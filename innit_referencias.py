from pathlib import Path
import pandas as pd

REFERENCIAS_XLSX = Path("Referencias.xlsx")

COLUMNS = [
    "ref_id","municipio","documento","ruta_documento","tipo_documento",
    "proyecto_id","proyecto_nombre","pagina_inicio","pagina_fin",
    "cita_texto","cita_resumen","justificacion","match_score","metodo_match",
    "modelo_llm","modelo_embeddings","extractor_version","fecha_extraccion"
]

def main():
    if REFERENCIAS_XLSX.exists():
        print("Referencias.xlsx ya existe. No lo toco.")
        return
    pd.DataFrame(columns=COLUMNS).to_excel(REFERENCIAS_XLSX, index=False, sheet_name="Referencias")
    print("OK. Creado Referencias.xlsx.")

if __name__ == "__main__":
    main()
