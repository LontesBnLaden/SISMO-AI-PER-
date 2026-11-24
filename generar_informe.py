"""Crea el informe PDF del proyecto y sus metricas reproducibles."""

from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sismo_model import train_and_evaluate

OUTPUT = ROOT / "informe" / "SISMO_AI_informe.pdf"
NAVY = colors.HexColor("#101B2D")
MINT = colors.HexColor("#B7F36B")
BLUE = colors.HexColor("#5969EA")
PAPER = colors.HexColor("#F3F4EF")
MUTED = colors.HexColor("#657185")
PALETTE = [colors.HexColor("#70B7A2"), colors.HexColor("#E7B85B"), colors.HexColor("#E97961")]


class Bars(Flowable):
    def __init__(self, values, labels, width=165 * mm, height=48 * mm, maximum=1):
        super().__init__()
        self.values, self.labels = values, labels
        self.width, self.height, self.maximum = width, height, maximum

    def draw(self):
        c = self.canv
        chart_left, chart_bottom, chart_height = 24 * mm, 8 * mm, self.height - 17 * mm
        chart_width = self.width - 30 * mm
        c.setStrokeColor(colors.HexColor("#D8DDD5"))
        c.line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height)
        c.line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom)
        n = len(self.values)
        slot = chart_width / max(n, 1)
        bar_width = min(19 * mm, slot * 0.54)
        for i, (value, label) in enumerate(zip(self.values, self.labels)):
            height = chart_height * value / self.maximum
            x = chart_left + i * slot + (slot - bar_width) / 2
            c.setFillColor(PALETTE[i % len(PALETTE)])
            c.roundRect(x, chart_bottom, bar_width, height, 2 * mm, fill=1, stroke=0)
            c.setFillColor(NAVY)
            c.setFont("Helvetica-Bold", 9)
            c.drawCentredString(x + bar_width / 2, chart_bottom + height + 3, f"{value:.0%}")
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 8)
            c.drawCentredString(x + bar_width / 2, chart_bottom - 12, label)


class Confusion(Flowable):
    def __init__(self, matrix, labels, width=115 * mm, height=63 * mm):
        super().__init__()
        self.matrix, self.labels, self.width, self.height = matrix, labels, width, height

    def draw(self):
        c = self.canv
        n = len(self.labels)
        cell = min(22 * mm, (self.height - 13 * mm) / max(n, 1), (self.width - 35 * mm) / max(n, 1))
        left, bottom = 31 * mm, 8 * mm
        maximum = max(max(row) for row in self.matrix) or 1
        for row in range(n):
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 8)
            c.drawRightString(left - 3, bottom + (n - row - 1) * cell + cell / 2 - 3, self.labels[row])
            for col in range(n):
                intensity = self.matrix[row][col] / maximum
                c.setFillColor(colors.Color(0.35 + (0.65 - 0.35) * intensity, 0.41 + (0.69 - 0.41) * intensity, 0.92 + (0.12 - 0.92) * intensity))
                x, y = left + col * cell, bottom + (n - row - 1) * cell
                c.roundRect(x, y, cell - 1.5, cell - 1.5, 2, fill=1, stroke=0)
                c.setFillColor(colors.white if intensity > 0.48 else NAVY)
                c.setFont("Helvetica-Bold", 10)
                c.drawCentredString(x + (cell - 1.5) / 2, y + cell / 2 - 4, str(self.matrix[row][col]))
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawCentredString(left + n * cell / 2, bottom - 8, "Predicción →")
        c.saveState()
        c.translate(5 * mm, bottom + n * cell / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "Etiqueta verdadera")
        c.restoreState()


def page_chrome(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 7 * mm, width, 7 * mm, fill=1, stroke=0)
    canvas.setFillColor(MINT)
    canvas.rect(0, height - 7 * mm, 47 * mm, 7 * mm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(20 * mm, 12 * mm, "SISMO AI · LABORATORIO EDUCATIVO · DATOS SINTÉTICOS")
    canvas.drawRightString(width - 20 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    result = train_and_evaluate()
    frame = result["data"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm, topMargin=17 * mm, bottomMargin=20 * mm, title="SISMO AI | Informe")
    doc.addPageTemplates([PageTemplate(id="normal", frames=Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="content"), onPage=page_chrome)])
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("HeroTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=30, leading=33, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8),
        "h1": ParagraphStyle("HeadOne", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=NAVY, spaceBefore=8, spaceAfter=8),
        "h2": ParagraphStyle("HeadTwo", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=BLUE, spaceBefore=6, spaceAfter=5),
        "body": ParagraphStyle("BodyTextX", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4, leading=14, textColor=NAVY, spaceAfter=7),
        "small": ParagraphStyle("SmallX", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED, spaceAfter=5),
        "cover": ParagraphStyle("Cover", parent=base["BodyText"], fontName="Helvetica", fontSize=12, leading=17, textColor=MUTED, spaceAfter=12),
        "white": ParagraphStyle("White", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=15, textColor=colors.white),
    }

    story = [
        Spacer(1, 21 * mm),
        Paragraph("SISMO AI", ParagraphStyle("brand", parent=styles["small"], textColor=BLUE, fontName="Helvetica-Bold", fontSize=10, leading=14)),
        Paragraph("Leer el movimiento.<br/>Entender el impacto.", styles["title"]),
        Paragraph("Informe de proyecto · Clasificación didáctica de escenarios sísmicos con K-Nearest Neighbors", styles["cover"]),
    ]
    cover_box = Table([[Paragraph("LABORATORIO DE DATOS · HUÁNUCO / PERÚ<br/><br/>Proyecto original de aprendizaje automático<br/>Interfaz, dataset y evaluación experimental", styles["white"])]], colWidths=[170 * mm], rowHeights=[40 * mm])
    cover_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("BOX", (0, 0), (-1, -1), 0, NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 10 * mm), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.extend([cover_box, Spacer(1, 13 * mm), Paragraph("Resumen", styles["h1"])])
    story.append(Paragraph(
        f"Se implementó un dashboard web con simulador KNN, mapa geográfico de escenarios, explorador del dataset y evaluación reproducible. "
        f"El CSV contiene <b>{len(frame)} filas sintéticas</b>, agrupadas en {frame['evento_id'].nunique()} eventos simulados y {frame['localidad'].nunique()} localidades. "
        "El objetivo es explicar un flujo de clasificación y sus límites, no representar datos observados ni anticipar terremotos.",
        styles["body"],
    ))
    notice = Table([[Paragraph("<b>AVISO</b>  Datos y etiquetas 100 % sintéticos. No son registros del IGP/CENSIS/USGS, no son intensidades Mercalli observadas y no deben usarse como alerta ni para decisiones de seguridad.", styles["body"])]], colWidths=[170 * mm])
    notice.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF2D7")), ("LINEBEFORE", (0, 0), (0, 0), 4, colors.HexColor("#E7B85B")), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7)]))
    story.extend([notice, Spacer(1, 7 * mm), Paragraph("01 · Diseño experimental", styles["h1"])])
    for text in [
        "<b>Variables:</b> magnitud, profundidad y distancia epicentral; etiqueta didáctica en tres clases (I-IV, V-VI, VII-XII).",
        "<b>Separación:</b> 80 % entrenamiento / 20 % prueba, agrupada por evento. Ningún evento se comparte entre conjuntos.",
        "<b>Selección:</b> K = 3, 5, 7 y 9 comparados por F1 macro con GroupKFold, solo sobre entrenamiento.",
        "<b>Preprocesamiento:</b> StandardScaler dentro de Pipeline; escalado ajustado nuevamente dentro de cada pliegue.",
        "<b>Evaluación:</b> métricas calculadas sobre eventos retenidos. Las etiquetas se generaron con una fórmula pedagógica artificial.",
    ]:
        story.append(Paragraph("• " + text, styles["body"]))
    story.append(Paragraph("02 · Dataset y procedencia", styles["h1"]))
    source_data = [
        ["Elemento", "Valor / significado"],
        ["Registros", f"{len(frame)} evento-localidad sintéticos"],
        ["Eventos", f"{frame['evento_id'].nunique()} grupos identificables (SYN-###)"],
        ["Localidades", f"{frame['localidad'].nunique()} localidades con nombres y coordenadas aproximadas"],
        ["Etiqueta", "Intensidad sintética; no es una observación en escala Mercalli"],
        ["Procedencia", "SINTETICO_EDUCATIVO en cada fila del CSV"],
    ]
    table = Table(source_data, colWidths=[39 * mm, 131 * mm], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 11), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D8DDD5")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.extend([table, Spacer(1, 4 * mm), Paragraph("El generador es reproducible (semilla fija) y se incluye como código fuente. Las localidades sirven como contexto cartográfico; no implican distancias geodésicas reales de las observaciones.", styles["small"])])

    story.append(Paragraph("03 · Resultados experimentales", styles["h1"]))
    metric_data = [["Exactitud", "Precisión macro", "Recall macro", "F1 macro"], [
        f"{result['metrics']['exactitud']:.1%}", f"{result['metrics']['precision_macro']:.1%}",
        f"{result['metrics']['recall_macro']:.1%}", f"{result['metrics']['f1_macro']:.1%}",
    ]]
    metrics_table = Table(metric_data, colWidths=[42.5 * mm] * 4, rowHeights=[10 * mm, 14 * mm])
    metrics_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("TEXTCOLOR", (0, 0), (-1, -1), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica"), ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 8), ("FONTSIZE", (0, 1), (-1, 1), 16), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.extend([Paragraph(f"K óptimo: {result['best_k']} · F1 macro promedio en CV agrupada: {result['cv_scores'][result['best_k']]:.1%}", styles["body"]), metrics_table, Spacer(1, 7 * mm), Paragraph("Comparación de K (F1 macro)", styles["h2"]), Bars([result["cv_scores"][k] for k in result["cv_scores"]], [f"K={k}" for k in result["cv_scores"]])])
    story.append(Paragraph("La prueba incluye "
        f"{result['test']['evento_id'].nunique()} eventos no vistos ({len(result['test'])} filas). "
        "Los valores describen únicamente la coherencia con el generador sintético; no son estimaciones de rendimiento en sismos reales.",
        styles["small"]))
    story.append(Paragraph("Matriz de confusión · prueba", styles["h2"]))
    story.append(Confusion(result["confusion"].tolist(), result["labels"]))
    story.append(Paragraph("Filas: etiqueta generada verdadera; columnas: clase asignada por el clasificador.", styles["small"]))

    story.append(Paragraph("04 · Producto e interpretación", styles["h1"]))
    for text in [
        "<b>Dashboard:</b> resumen del conjunto, mapa de epicentros simulados, distribución de clases y curva de selección K.",
        "<b>Simulador:</b> permite variar magnitud, profundidad, distancia y K; enseña la clase y votos de los vecinos reales del conjunto de entrenamiento.",
        "<b>Evaluación:</b> presenta exactitud, precisión, recall, F1 macro, matriz de confusión y validación agrupada.",
        "<b>Fuentes sugeridas para futuro:</b> IGP, CENSIS y USGS. Se debe verificar que el catálogo incluya etiquetas de intensidad observada, además de documentar unidades, permisos y procedencia.",
        "<b>Conclusión:</b> la aplicación demuestra un pipeline KNN completo y previene fuga por evento. Para conclusiones científicas, el CSV debe reemplazarse por registros etiquetados observados y el modelo necesita validación externa.",
    ]:
        story.append(Paragraph("• " + text, styles["body"]))
    story.append(Paragraph("Preguntas para la sustentación", styles["h2"]))
    for question in [
        "¿Por qué el escalado cambia qué vecinos son cercanos cuando las variables tienen unidades distintas?",
        "¿Qué diferencia hay entre magnitud e intensidad, y cuál de ellas representa la etiqueta del prototipo?",
        "¿Por qué agrupar la división por evento y comparar K solo en entrenamiento?",
        "¿Qué evidencia se necesitaría para reemplazar el dataset sintético por uno real?",
    ]:
        story.append(Paragraph("– " + question, styles["body"]))
    doc.build(story)
    print(f"Informe PDF generado: {OUTPUT}")


if __name__ == "__main__":
    main()
