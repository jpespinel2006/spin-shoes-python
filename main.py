from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import psycopg2
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ai import responder_inteligente, indexar_datos
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


class ChatRequest(BaseModel):
    pregunta: str


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "6543"),
        sslmode="require",
        options="-c search_path=public"
    )


@app.post("/chat")
def chat(data: ChatRequest):
    pregunta = data.pregunta
    respuesta = responder_inteligente(pregunta)

    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO public.ai_logs (pregunta, respuesta) VALUES (%s, %s)",
            (pregunta, respuesta)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("Error guardando en ai_logs:", e)

    return {"respuesta": respuesta}


@app.get("/")
def home():
    return {"message": "API funcionando"}


@app.get("/analytics")
def analytics():
    try:
        conn = get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM public.orders WHERE status = 'activo'")
        activos = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM public.orders WHERE status = 'produccion'")
        produccion = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM public.orders WHERE status = 'completado'")
        completados = cursor.fetchone()[0]

        conn.close()

    except Exception as e:
        print("ERROR BD:", e)
        activos = 0
        produccion = 0
        completados = 0

    estados = ["Activos", "Producción", "Completados"]
    valores = [activos, produccion, completados]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3B82F6", "#F59E0B", "#10B981"]
    ax.bar(estados, valores, color=colors, width=0.5)
    ax.set_title("Estado de Pedidos", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cantidad")
    for i, v in enumerate(valores):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("chart.png", dpi=100)
    plt.close()

    base_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")

    return {
        "activos": activos,
        "produccion": produccion,
        "completados": completados,
        "chart": f"{base_url}/static/chart.png"
    }


@app.post("/admin/reindex")
def reindex():
    try:
        total = indexar_datos()
        return {"ok": True, "documentos_indexados": total}
    except Exception as e:
        return {"ok": False, "error": str(e)}
