"""Genera un conjunto reproducible, totalmente sintetico y etiquetado."""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20261123
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "sismos_sinteticos.csv"
SITES = [
    ("Huanuco", -9.93, -76.24), ("Lima", -12.05, -77.04),
    ("Huaraz", -9.53, -77.53), ("Ica", -14.07, -75.73),
    ("Arequipa", -16.40, -71.54), ("Cusco", -13.52, -71.97),
    ("Ayacucho", -13.16, -74.22), ("Pucallpa", -8.38, -74.53),
    ("Tingo Maria", -9.29, -76.00), ("Tarma", -11.42, -75.69),
    ("Cerro de Pasco", -10.68, -76.26), ("Chiclayo", -6.77, -79.84),
    ("Trujillo", -8.11, -79.03), ("Piura", -5.19, -80.63),
    ("Moquegua", -17.20, -70.93), ("Tacna", -18.01, -70.25),
    ("Abancay", -13.64, -72.88), ("Chachapoyas", -6.23, -77.87),
]


def main():
    rng = random.Random(SEED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "evento_id", "fecha", "latitud", "longitud", "localidad", "magnitud",
        "profundidad_km", "distancia_km", "intensidad", "procedencia",
    ]
    rows = []
    start = date(2000, 1, 1)
    for event_no in range(1, 37):
        event_id = f"SYN-{event_no:03d}"
        event_date = start + timedelta(days=rng.randint(0, 9400))
        epicenter_lat = round(rng.uniform(-18.2, -0.8), 4)
        epicenter_lon = round(rng.uniform(-81.0, -68.5), 4)
        magnitude = rng.uniform(3.0, 7.8)
        depth = rng.uniform(8, 190)
        for locality, _, _ in SITES:
            distance = rng.uniform(8, 720)
            # Mecanismo didactico artificial; no representa Mercalli observada.
            score = magnitude - 0.55 * math.log10(distance + 1) - 0.003 * depth
            score += rng.gauss(0, 0.24)
            intensity = "I-IV" if score < 4.0 else ("V-VI" if score < 5.45 else "VII-XII")
            rows.append(
                {
                    "evento_id": event_id,
                    "fecha": event_date.isoformat(),
                    "latitud": f"{epicenter_lat:.4f}",
                    "longitud": f"{epicenter_lon:.4f}",
                    "localidad": locality,
                    "magnitud": f"{magnitude:.2f}",
                    "profundidad_km": f"{depth:.2f}",
                    "distancia_km": f"{distance:.2f}",
                    "intensidad": intensity,
                    "procedencia": "SINTETICO_EDUCATIVO",
                }
            )
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dataset generado: {len(rows)} filas, {OUTPUT}")


if __name__ == "__main__":
    main()
