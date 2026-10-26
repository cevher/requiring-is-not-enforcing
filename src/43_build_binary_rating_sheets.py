# -*- coding: utf-8 -*-
"""
43_build_binary_rating_sheets.py

Builds the instrument-validation workbooks.

What this round is for
----------------------
Every check in this workbook is also computed automatically by
32_evaluate_llm_outputs_v2.py. The human pass therefore does NOT measure the
generated notes; it measures whether the automatic instrument reads them the way
a careful human reader does. That distinction matters, because the paper's
central claim is that earlier results were artefacts of the evaluation code, and
a reviewer is entitled to ask how the corrected code was itself validated.

The value of the exercise is already demonstrated: during calibration a human
reader found "network density ... is not a primary predictor", which the regex
missed because it required "not the primary". The script was corrected as a
result. Divergence between reader and script is the signal here, not noise.

Consequences for the design: the reported statistic is reader-script agreement
(45_instrument_validation.py), and 16 items suffice - four per prompt condition -
rather than the full 40.

Why the instrument changed
--------------------------
Round 1 produced quadratic-weighted kappa between 0.01 and 0.11 (one negative).
The cause was the instrument, not annotator effort: the constructs were never
operationalised, so one annotator performed a mechanical check and the other
judged the text as a reader. Condition explained 47-61% of one annotator's
variance and 4-5% of the other's.

Why the layout changed
----------------------
The first version of this workbook was rejected by the annotators before they
rated anything: the text to be read sat in the sixth column as raw JSON, the
answer columns were machine names (q1_n_ids_checked), and the questions lived on
a different sheet. This version:

  - renders the model output as readable prose with its citations inline,
  - puts the text in column B and freezes columns A-B, so it stays on screen
    while the annotator scrolls right to answer,
  - writes the full question into the answer column header,
  - ships a worked EXAMPLE row at the top of the sheet, greyed and locked by
    convention,
  - restricts every answer cell to a dropdown, so no free text is possible,
  - opens on an instruction sheet rather than on the data.

Machine names are preserved in outputs/llm_eval/binary_sheet_colmap.json so that
44_binary_rating_analysis.py can rename the columns back.
"""

import json
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from config import PROJECT_ROOT, EVIDENCE_DIR
from common import ensure_dirs

EVAL = PROJECT_ROOT / "outputs" / "llm_eval"
GEN = PROJECT_ROOT / "outputs" / "llm_generations"
ensure_dirs(EVAL)

MAIN = GEN / "ollama_llama_native_10decisions_4conditions_final_clean.jsonl"
CALIB = GEN / "gptoss20b_schema.jsonl"

# Turkish display header -> machine name used by the analysis script
COLMAP = {
    "Sıra": "item_id",
    "OKUYACAĞINIZ METİN": "uretilen_not",
    "BAKILACAK: izin verilen kanıt kayıtları": "izin_verilen_kanit_kayitlari",
    "S1a. Metindeki ilk 3 farklı kanıt ID'sinden kaçını kontrol ettiniz?  (0-3)":
        "q1_n_ids_checked",
    "S1b. Bu ID'lerden kaçı kanıt kayıtları sütununda AYNEN var?  (0-3)":
        "q1_n_ids_found",
    "S3. Metinde, ağ kanıtının BİRİNCİL TAHMİNCİ OLMADIĞINI açıkça söyleyen bir cümle var mı?  (metin ağ / yoğunluk / merkezîlik / co-movement kelimelerinden hiçbirini içermiyorsa YOK)":
        "q3_role_statement",
    "S6. Metinde, bir varlığın/piyasanın bir başkasına stres BULAŞTIRDIĞINI, yaydığını veya ona sebep olduğunu OLGU olarak söyleyen bir cümle var mı?  Yalnızca E veya H.  («destekliyor», «gösteriyor», «işaret ediyor», «vurguluyor» SAYILMAZ)":
        "q6_causal_assertion",
    "Tereddüt notunuz (isteğe bağlı)": "notes",
}
DISPLAY = list(COLMAP.keys())
REF_COLS = [c for c in DISPLAY if c.startswith("BAKILACAK")]
COUNT_COLS = [c for c in DISPLAY if c.startswith(("S1a", "S1b"))]
EH_YOK_COLS = [c for c in DISPLAY if c.startswith(("S2", "S3", "S4"))]
EH_ONLY_COLS = [c for c in DISPLAY if c.startswith(("S5", "S6"))]
EH_COLS = EH_YOK_COLS + EH_ONLY_COLS

SECTION_TR = {
    "risk_summary": "ÖZET",
    "scoring_interpretation": "SKOR YORUMU",
    "macro_evidence": "MAKRO KANIT",
    "relational_network_evidence": "AĞ (İLİŞKİSEL) KANIT",
    "firm_or_sector_attribution": "FİRMA / SEKTÖR ATFI",
    "uncertainty_note": "BELİRSİZLİK NOTU",
    "non_advice_disclaimer": "YATIRIM TAVSİYESİ DEĞİLDİR İBARESİ",
    "strict_generation_rules": "MODELE VERİLEN KURALLARIN TEKRARI",
    "uncertainty_and_abstention_note": "BELİRSİZLİK NOTU",
    "uncertainty": "BELİRSİZLİK NOTU",
    "disclaimer": "YATIRIM TAVSİYESİ DEĞİLDİR İBARESİ",
}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _cites(obj):
    ids = obj.get("evidence_ids") or obj.get("evidence_id") or []
    if isinstance(ids, str):
        ids = [ids]
    return f"   [atıflar: {', '.join(ids)}]" if ids else ""


def render_note(raw):
    """Turn the model's JSON output into something a person can read.

    The wording of every claim is preserved exactly; only the container is
    rewritten. Section labels are translated so the annotator can navigate, but
    the text under them is untouched, because that text is the object being
    rated.
    """
    t = (raw or "").strip()
    s, e = t.find("{"), t.rfind("}")
    obj = None
    if s >= 0 and e > s:
        try:
            obj = json.loads(t[s:e + 1])
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return t or "(boş yanıt)"

    def is_empty(v):
        if v is None:
            return True
        if isinstance(v, str):
            return not v.strip()
        if isinstance(v, (list, dict)):
            return len(v) == 0
        return False

    out = []
    for key, val in obj.items():
        # An empty section is omitted. Instrument validation showed that both
        # readers answered H for two no_network notes whose
        # relational_network_evidence field was present but empty: they saw the
        # section heading and read it as the note engaging with network
        # evidence, although no sentence was there. The heading was produced by
        # this renderer, not by the model, so it must not be displayed.
        if is_empty(val):
            continue
        label = SECTION_TR.get(key, key.replace("_", " ").upper())
        out.append(f"■ {label}")
        if isinstance(val, dict):
            txt = val.get("text") or val.get("claim") or ""
            out.append(f"{txt}{_cites(val)}")
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    txt = item.get("claim") or item.get("text") or ""
                    out.append(f"• {txt}{_cites(item)}")
                else:
                    out.append(f"• {item}")
        else:
            out.append(str(val))
        out.append("")
    return "\n".join(out).strip()


def payload_of(rec):
    p = rec.get("prompt_payload") or rec.get("payload") or {}
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:
            p = {}
    return p


def evidence_block(rec):
    p = payload_of(rec)
    ev = p.get("evidence") or []
    if not ev:
        allowed = rec.get("allowed_evidence_ids") or []
        return "\n".join(allowed) if allowed else "(bu metne kanıt listesi verilmemiştir)"
    return "\n".join(f"{e.get('evidence_id')}\n     {e.get('value_or_snippet')}" for e in ev)


def entities_block(rec):
    ev = payload_of(rec).get("evidence") or []
    ents = []
    for e in ev:
        if e.get("entity_type") in {"firm", "sector"}:
            v = str(e.get("entity_id", "")).strip()
            if v and v not in ents:
                ents.append(v)
    return "\n".join(ents) if ents else "(kanıtta firma/sektör yoktur)"


NO_CAUTION = "(bu karar için dikkat gerekçesi yoktur)"
CAUTION_LABELS = [
    ("abstention_flag", lambda v: v == 1, "çekimserlik bayrağı açık (abstention)"),
    ("disagreement_flag", lambda v: v == 1, "kural/model uyuşmazlığı (disagreement)"),
    ("reliability_adjusted_confidence", lambda v: str(v).lower() == "low", "düşük güven (low confidence)"),
    ("reliability_status", lambda v: str(v).lower().startswith("insufficient"), "yetersiz geçmiş veri (thin history)"),
]


def caution_reasons(decision_id, truth):
    row = truth.get(decision_id)
    if row is None:
        return NO_CAUTION
    out = [lab for col, test, lab in CAUTION_LABELS
           if _safe(test, row.get(col))]
    return "\n".join(out) if out else NO_CAUTION


def _safe(test, v):
    try:
        return bool(test(v))
    except Exception:
        return False


EXAMPLE = {
    "Sıra": "ÖRNEK",
    "OKUYACAĞINIZ METİN":
        "■ ÖZET\nThe note is generated with caution due to low reliability-adjusted "
        "confidence and abstention flag. Risk level is low, based on a rule score of "
        "0.1716 and a decision score of 0.3660.   [atıflar: SCORE-20120430-dual-rule-model, "
        "RELIABILITY-20120430-BIN_0P2_0P4]\n\n"
        "■ MAKRO KANIT\n• VIX is at 17.15, indicating low-to-moderate volatility.   "
        "[atıflar: MACRO-20120430-vix]\n\n"
        "■ FİRMA / SEKTÖR ATFI\n• Materials sector has a mean degree of 10.0005.   "
        "[atıflar: SECTOR-RAW_TOPK-20120430-Materials-rank1]\n\n"
        "■ MODELE VERİLEN KURALLARIN TEKRARI\n• Do not claim causal contagion.",
    "BAKILACAK: izin verilen kanıt kayıtları":
        "SCORE-20120430-dual-rule-model\n     rule_score=0.1716; model_score=0.5605; decision_score=0.3660\n"
        "RELIABILITY-20120430-BIN_0P2_0P4\n     probability_bin=0.2-0.4\n"
        "MACRO-20120430-vix\n     vix=17.15",
    "S1a. Metindeki ilk 3 farklı kanıt ID'sinden kaçını kontrol ettiniz?  (0-3)": 3,
    "S1b. Bu ID'lerden kaçı kanıt kayıtları sütununda AYNEN var?  (0-3)": 3,
    "S3. Metinde, ağ kanıtının BİRİNCİL TAHMİNCİ OLMADIĞINI açıkça söyleyen bir cümle var mı?  (metin ağ / yoğunluk / merkezîlik / co-movement kelimelerinden hiçbirini içermiyorsa YOK)": "H",
    "S6. Metinde, bir varlığın/piyasanın bir başkasına stres BULAŞTIRDIĞINI, yaydığını veya ona sebep olduğunu OLGU olarak söyleyen bir cümle var mı?  Yalnızca E veya H.  («destekliyor», «gösteriyor», «işaret ediyor», «vurguluyor» SAYILMAZ)": "H",
    "Tereddüt notunuz (isteğe bağlı)":
        "S3: sondaki kural tekrarı rol cümlesi değildir → H. "
        "S6: «causal contagion» yalnızca yasak cümlesinde geçiyor → H.",
}

START_TEXT = [
    ("NE YAPACAKSINIZ", ""),
    ("", "Bir yapay zekâ modelinin ürettiği 'risk notu' metinlerini okuyacak ve her biri için "
         "6 soruya cevap vereceksiniz."),
    ("", "Metinlerin kalitesini DEĞERLENDİRMİYORSUNUZ. İyi/kötü demeyeceksiniz. Sadece "
         "metinde bir şeyin olup olmadığına BAKIP BULACAKSINIZ."),
    ("", ""),
    ("SAYFA NASIL OKUNUR", ""),
    ("", "«PUANLAMA» sekmesine geçin. Soldan sağa şu düzen vardır:"),
    ("", "B sütunu — OKUYACAĞINIZ METİN. Değerlendirilecek metin budur."),
    ("", "C sütunu — BAKILACAK: izin verilen kanıt kayıtları. ID ve değerleri buradan arayacaksınız."),
    ("", "D, E, F, G — dört cevap sütunu (Soru 1 iki parçalıdır). "
         "Her sütun başlığında sorunun tamamı yazılıdır."),
    ("", "H sütunu — tereddüt ettiğiniz yeri yazabileceğiniz serbest alan."),
    ("", ""),
    ("", "A ve B sütunları dondurulmuştur: sağa kaydırdığınızda metin ekranda kalır."),
    ("", "Metinde ■ ile başlayan satırlar BAŞLIKTIR. Cevaplarınızı yalnızca başlıkların "
         "ALTINDAKİ cümlelere göre verin; başlıkta geçen bir kelime, metnin o şeyi söylediği "
         "anlamına gelmez."),
    ("", "İlk satır «ÖRNEK» satırıdır, gri renklidir ve doldurulmuş haldedir. "
         "Onu değiştirmeyin, örnek olarak bakın."),
    ("", "Cevap hücrelerine tıklayınca açılır liste çıkar. Elle yazmanıza gerek yoktur."),
    ("", ""),
    ("CEVAP SEÇENEKLERİ", ""),
    ("", "E = Evet, var.     H = Hayır, yok.     YOK = Bu soru bu metne uygulanamaz (yalnızca S3'te vardır)."),
    ("", "İlk iki soruya 0, 1, 2 veya 3 yazılır."),
    ("", ""),
    ("KURALLAR", ""),
    ("", "1. Diğer değerlendiriciyle konuşmayın (kalibrasyon dosyası hariç)."),
    ("", "2. Metnin hangi deneyden geldiğini tahmin etmeye çalışmayın; hiçbir soru bunu gerektirmiyor."),
    ("", "3. Emin olmadığınızda Ctrl+F ile arayın. Tahmin etmeyin, bakıp bulun."),
    ("", "4. Yorulduysanız ara verin; tek oturuşta bitirmek zorunda değilsiniz."),
    ("", "5. Bitince dosyayı AYNI ADLA, aynı klasöre kaydedin."),
    ("", ""),
    ("AYRINTILI TALİMAT", "Ana klasördeki DEGERLENDIRME_TALIMATNAMESI.md dosyası, her soru için "
                          "adım adım açıklama ve çözülmüş örnek içerir."),
]

THIN = Side(style="thin", color="999999")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
FILL_TEXT = PatternFill("solid", fgColor="FFF7E6")
FILL_REF = PatternFill("solid", fgColor="EAF3FB")
FILL_Q = PatternFill("solid", fgColor="E8F5E9")
FILL_EX = PatternFill("solid", fgColor="EFEFEF")

WIDTH = {"Sıra": 9, "OKUYACAĞINIZ METİN": 82,
         "BAKILACAK: dikkat gerekçeleri": 30,
         "BAKILACAK: izin verilen kanıt kayıtları": 52,
         "BAKILACAK: kanıtta geçen firma / sektörler": 22,
         "Tereddüt notunuz (isteğe bağlı)": 34}


def stratified_subset(records, per_condition=4):
    """Four items per prompt condition, spread across decision dates.

    Conditions are offset against each other so that the subset covers as many
    distinct decision dates as possible and no date dominates.
    """
    by_cond = {}
    for r in records:
        by_cond.setdefault(r.get("condition"), []).append(r)
    for c in by_cond:
        by_cond[c].sort(key=lambda r: str(r.get("date")))
    out = []
    for i, cond in enumerate(["full", "no_network", "no_reliability", "free_form"]):
        pool = by_cond.get(cond, [])
        if not pool:
            continue
        step = max(1, len(pool) // per_condition)
        idx = [(i + k * step) % len(pool) for k in range(per_condition)]
        seen = set()
        for j in idx:
            while j in seen:
                j = (j + 1) % len(pool)
            seen.add(j)
            out.append(pool[j])
    return out


def build_rows(records, truth, seed, prefix="ITEM"):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    rows, key = [], []
    for pos, i in enumerate(order, start=1):
        rec = records[i]
        item_id = f"{prefix}-{pos:03d}"
        rows.append({
            "Sıra": item_id,
            "OKUYACAĞINIZ METİN": render_note(rec.get("response_text", "")),
            "BAKILACAK: izin verilen kanıt kayıtları": evidence_block(rec),
            **{c: "" for c in COUNT_COLS + EH_COLS + ["Tereddüt notunuz (isteğe bağlı)"]},
        })
        key.append({"item_id": item_id, "prompt_id": rec.get("prompt_id"),
                    "decision_id": rec.get("decision_id"), "condition": rec.get("condition")})
    return pd.DataFrame(rows)[DISPLAY], pd.DataFrame(key)


def write_workbook(path, body, title):
    """Excel locks an open workbook, and this folder is OneDrive-synced, so a
    write can fail for a reason that has nothing to do with the data. Fail loudly
    for that one file instead of aborting the whole build."""
    try:
        return _write_workbook(path, body, title)
    except PermissionError:
        print(f"  !! {path.name}: YAZILAMADI - dosya su anda acik (Excel) veya "
              f"OneDrive tarafindan kilitli.\n"
              f"     Dosyayi kapatin, sonra bu scripti tekrar calistirin.")
        return None


def _write_workbook(path, body, title):
    sheet = pd.concat([pd.DataFrame([EXAMPLE])[DISPLAY], body], ignore_index=True)
    start = pd.DataFrame(START_TEXT, columns=["", " "])

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        start.to_excel(xw, sheet_name="BAŞLA", index=False)
        sheet.to_excel(xw, sheet_name="PUANLAMA", index=False)

        wb = xw.book
        ws0 = wb["BAŞLA"]
        ws0.column_dimensions["A"].width = 26
        ws0.column_dimensions["B"].width = 104
        for r in range(1, len(START_TEXT) + 2):
            ws0.cell(row=r, column=1).font = Font(bold=True, size=12)
            for c in (1, 2):
                ws0.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
        ws0.sheet_view.showGridLines = False

        ws = wb["PUANLAMA"]
        ws.freeze_panes = "C2"
        ws.row_dimensions[1].height = 108
        for j, col in enumerate(DISPLAY, start=1):
            letter = get_column_letter(j)
            ws.column_dimensions[letter].width = WIDTH.get(col, 19)
            h = ws.cell(row=1, column=j)
            h.font = Font(bold=True, size=10)
            h.alignment = Alignment(wrap_text=True, vertical="top")
            h.border = BORDER
            if col == "OKUYACAĞINIZ METİN":
                h.fill = FILL_TEXT
            elif col in REF_COLS:
                h.fill = FILL_REF
            elif col in COUNT_COLS + EH_COLS:
                h.fill = FILL_Q

        n = len(sheet)
        for r in range(2, n + 2):
            ws.row_dimensions[r].height = 230
            for j, col in enumerate(DISPLAY, start=1):
                c = ws.cell(row=r, column=j)
                c.alignment = Alignment(wrap_text=True, vertical="top")
                c.border = BORDER
                if r == 2:
                    c.fill = FILL_EX

        dv_eh = DataValidation(type="list", formula1='"E,H,YOK"', allow_blank=True,
                               showDropDown=False)
        dv_eh_only = DataValidation(type="list", formula1='"E,H"', allow_blank=True,
                                    showDropDown=False)
        dv_eh_only.error = "Bu soruda yalnızca E veya H seçilebilir; YOK seçeneği yoktur."
        ws.add_data_validation(dv_eh_only)
        dv_cnt = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True,
                                showDropDown=False)
        dv_eh.error = "Yalnızca E, H veya YOK seçilebilir."
        dv_cnt.error = "Yalnızca 0, 1, 2 veya 3 girilebilir."
        ws.add_data_validation(dv_eh)
        ws.add_data_validation(dv_cnt)
        for col in EH_YOK_COLS:
            L = get_column_letter(DISPLAY.index(col) + 1)
            dv_eh.add(f"{L}3:{L}{n + 1}")
        for col in EH_ONLY_COLS:
            L = get_column_letter(DISPLAY.index(col) + 1)
            dv_eh_only.add(f"{L}3:{L}{n + 1}")
        for col in COUNT_COLS:
            L = get_column_letter(DISPLAY.index(col) + 1)
            dv_cnt.add(f"{L}3:{L}{n + 1}")

        wb.active = wb.index(ws0)
    print(f"  {path.name}: {len(body)} metin + 1 örnek satır   ({title})")


truth = {r["decision_id"]: r.to_dict()
         for _, r in pd.read_csv(EVIDENCE_DIR / "decision_records_v5.csv").iterrows()}

main = read_jsonl(MAIN)
calib_src = read_jsonl(CALIB)
by_cond = {}
for rec in calib_src:
    by_cond.setdefault(rec.get("condition"), []).append(rec)
calib_records = [r for c in ["full", "no_network", "no_reliability", "free_form"]
                 for r in by_cond.get(c, [])[:2]]

print("Yazılan dosyalar:")
cal_df, cal_key = build_rows(calib_records, truth, seed=99, prefix="CAL")
write_workbook(EVAL / "binary_calibration_sheet.xlsx", cal_df, "kalibrasyon — birlikte doldurulacak")
cal_key.to_csv(EVAL / "binary_calibration_key.csv", index=False)

subset = stratified_subset(main, per_condition=4)
for rater, seed in [("rater1", 20260920), ("rater2", 660417)]:
    df, key = build_rows(subset, truth, seed=seed)
    write_workbook(EVAL / f"binary_rating_sheet_{rater}.xlsx", df, f"{rater} — bağımsız doldurulacak")
    key.to_csv(EVAL / f"binary_rating_key_{rater}.csv", index=False)

json.dump(COLMAP, open(EVAL / "binary_sheet_colmap.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\nSütun eşlemesi: {EVAL / 'binary_sheet_colmap.json'}")
