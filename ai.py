from sentence_transformers import SentenceTransformer
import psycopg2
import psycopg2.extras
import os
import json as _json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port":     os.getenv("DB_PORT", "6543"),
    "sslmode":  "require",
    "options":  "-c search_path=public"
}


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def indexar_datos():
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    documentos = []

    cur.execute("""
        SELECT id, status, cantidad, cliente, modelo, personalizacion, created_at
        FROM public.orders
        ORDER BY id DESC
        LIMIT 500
    """)
    pedidos = cur.fetchall()
    for row in pedidos:
        pers = {}
        if row['personalizacion']:
            try:
                pers = _json.loads(row['personalizacion']) if isinstance(row['personalizacion'], str) else row['personalizacion']
            except Exception:
                pass

        color    = pers.get('color_producto', pers.get('color', ''))
        suela    = pers.get('suela', '')
        tallas   = pers.get('tallas', {})
        nota     = pers.get('nota', '')

        tallas_str = ""
        if tallas:
            partes = [f"talla {t}: {c} pares" for t, c in tallas.items() if int(c or 0) > 0]
            if partes:
                tallas_str = ", " + ", ".join(partes)

        texto = (
            f"Pedido #{row['id']} — "
            f"cliente: {row['cliente']}, "
            f"referencia: {row['modelo']}, "
            f"estado: {row['status']}, "
            f"cantidad total: {row['cantidad']} pares"
            f"{tallas_str}"
            f"{f', color: {color}' if color else ''}"
            f"{f', suela: {suela}' if suela else ''}"
            f"{f', nota: {nota}' if nota else ''}."
        )
        documentos.append(("orders", texto))

    cur.execute("SELECT COUNT(*) FROM public.orders")
    total_pedidos = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM public.orders WHERE status = 'activo'")
    activos = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM public.orders WHERE status = 'produccion'")
    produccion = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM public.orders WHERE status = 'completado'")
    completados = cur.fetchone()[0]
    cur.execute("SELECT SUM(cantidad) FROM public.orders")
    total_pares = cur.fetchone()[0] or 0

    resumen_pedidos = (
        f"Resumen general de pedidos: hay {total_pedidos} pedidos en total. "
        f"{activos} pedidos están activos. "
        f"{produccion} pedidos están en producción. "
        f"{completados} pedidos están completados. "
        f"Total de pares: {total_pares}."
    )
    documentos.append(("orders_resumen", resumen_pedidos))

    cur.execute("SELECT id, nombre, nit, ciudad, tipo_cliente, telefono, email FROM public.clients LIMIT 200")
    for row in cur.fetchall():
        texto = (
            f"Cliente {row['nombre']} (ID {row['id']}), "
            f"NIT: {row['nit']}, "
            f"ciudad: {row['ciudad']}, "
            f"tipo: {row['tipo_cliente']}, "
            f"teléfono: {row['telefono']}, "
            f"email: {row['email']}."
        )
        documentos.append(("clients", texto))

    cur.execute("SELECT referencia, descripcion, precio, stock FROM public.catalog LIMIT 200")
    for row in cur.fetchall():
        texto = (
            f"Referencia {row['referencia']}: {row['descripcion']}, "
            f"precio ${row['precio']}, "
            f"stock: {row['stock']} unidades."
        )
        documentos.append(("catalog", texto))

    cur.execute("DELETE FROM public.documentos_rag")
    for fuente, contenido in documentos:
        emb = model.encode(contenido).tolist()
        cur.execute(
            "INSERT INTO public.documentos_rag (contenido, embedding, fuente) VALUES (%s, %s::vector, %s)",
            (contenido, str(emb), fuente)
        )

    conn.commit()
    cur.close()
    conn.close()
    return len(documentos)


def buscar_en_cache(pregunta: str):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT respuesta FROM public.ai_logs WHERE LOWER(pregunta) = LOWER(%s) LIMIT 1",
            (pregunta,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return None


def buscar_contexto(emb: list, k: int = 5) -> str:
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT contenido
            FROM public.documentos_rag
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (str(emb), k))
        resultados = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return "\n".join(resultados)
    except Exception as e:
        print("Error buscando contexto:", e)
        return ""


def responder_inteligente(pregunta: str) -> str:
    cached = buscar_en_cache(pregunta)
    if cached:
        return cached

    emb = model.encode(pregunta).tolist()
    contexto = buscar_contexto(emb, k=5)

    if not contexto.strip():
        return "No encontré información relevante en el sistema para responder esa consulta."

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres el asistente inteligente de Spin Shoes SAS, una empresa de fabricación de calzado. "
                        "Tienes acceso a información de pedidos, clientes y catálogo. "
                        "Responde ÚNICAMENTE con la información del contexto proporcionado. "
                        "Sé directo y conciso. No inventes datos."
                    )
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
                }
            ],
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error OpenAI:", e)
        return "No pude procesar tu consulta en este momento."
