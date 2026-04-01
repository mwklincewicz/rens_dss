# src/harmonization.py

FUSIE_MAPPING = {
    # voorbeeldstructuur: oude_code: nieuwe_code

    # Meierijstad (2017)
    855: 1961,  # Schijndel -> Meierijstad
    794: 1961,  # Sint-Oedenrode -> Meierijstad
    840: 1961,  # Veghel -> Meierijstad

    # Eemsdelta (2021)
    610: 1994,  # Appingedam -> Eemsdelta
    15: 1994,   # Delfzijl -> Eemsdelta
    18: 1994,   # Loppersum -> Eemsdelta

    # Dijk en Waard (2022)
    1598: 1978,  # Heerhugowaard -> Dijk en Waard
    1916: 1978,  # Langedijk -> Dijk en Waard

    # Land van Cuijk (2022)
    828: 1999,  # Boxmeer -> Land van Cuijk
    785: 1999,  # Cuijk -> Land van Cuijk
    786: 1999,  # Grave -> Land van Cuijk
    787: 1999,  # Mill en Sint Hubert -> Land van Cuijk
    1676: 1999, # Sint Anthonis -> Land van Cuijk

    # etc. → hier kun je later makkelijk aan toevoegen
}

# Haaren is speciaal: wordt opgesplitst
SPLIT_MAPPING = {
    # oude_code: {nieuwe_code: gewicht, ...}
    788: {      # Haaren
        756: 0.134,  # Boxtel
        824: 0.357,  # Oisterwijk
        855: 0.125,  # Tilburg
        865: 0.384,  # Vught
    }
}

import pandas as pd

def harmonize_municipalities(
    df: pd.DataFrame,
    code_col: str,
    year_col: str | None = None,
    value_cols: list[str] | None = None,
    fusie_mapping: dict[int, int] | None = None,
    split_mapping: dict[int, dict[int, float]] | None = None,
    new_code_col: str = "GemeenteCode_harmonized",
) -> pd.DataFrame:
    """
    Harmoniseer gemeentecodes over de tijd:
    - past fusies toe (oude -> nieuwe code)
    - verwerkt splitsingen (zoals Haaren) met gewichten
    - werkt voor elke dataset (SES, bevolking, oppervlakte, etc.)
    """

    fusie_mapping = fusie_mapping or {}
    split_mapping = split_mapping or {}

    df = df.copy()

    # zorg dat codes ints zijn
    df[code_col] = df[code_col].astype(int)

    # start met nieuwe codekolom
    df[new_code_col] = df[code_col].replace(fusie_mapping)

    # voor codes die niet in fusie_mapping staan: behoud origineel
    mask_missing = df[new_code_col].isna()
    df.loc[mask_missing, new_code_col] = df.loc[mask_missing, code_col]

    # splitsingen verwerken (zoals Haaren)
    if split_mapping and value_cols is not None:
        rows = []

        for _, row in df.iterrows():
            code = int(row[code_col])

            if code in split_mapping:
                # verdeel over nieuwe gemeenten
                for new_code, weight in split_mapping[code].items():
                    new_row = row.copy()
                    new_row[new_code_col] = new_code
                    for vc in value_cols:
                        new_row[vc] = new_row[vc] * weight
                    rows.append(new_row)
            else:
                rows.append(row)

        df = pd.DataFrame(rows)

    return df