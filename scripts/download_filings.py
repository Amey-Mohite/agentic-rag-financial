"""
download_filings.py — download a few public SEC filings into ./data as PDFs.

Uses SEC EDGAR's public endpoints. SEC requires a descriptive User-Agent with contact info; set
SEC_USER_AGENT in your environment (e.g. "Your Name your@email.com").

Usage:
    python scripts/download_filings.py
    python scripts/download_filings.py --tickers AAPL MSFT --forms 10-K 10-Q
"""

from __future__ import annotations

import os
import argparse
import time
import json
import urllib.request


# A small default set chosen for VARIETY (dense 10-Ks + a quarterly), so retrieval does real work.
DEFAULT_TICKERS = ["AAPL", "MSFT"]
DEFAULT_FORMS = ["10-K"]

UA = os.environ.get("SEC_USER_AGENT", "agentic-rag-demo contact@example.com")
# Ask for identity (uncompressed). We still gunzip defensively below in case SEC compresses anyway.
HEADERS = {"User-Agent": UA, "Accept-Encoding": "identity"}


def _get(url: str) -> bytes:
    """HTTP GET with the SEC-required User-Agent, transparently decompressing gzip if present.

    WHY the gunzip guard: SEC sometimes returns gzip-compressed bodies (magic bytes 0x1f 0x8b)
    regardless of the request headers. urllib does NOT auto-decompress, so we detect and inflate it
    ourselves — otherwise json.loads sees raw gzip bytes and dies with a utf-8 decode error.
    PARAM url: the URL to fetch.
    RETURNS: decompressed response bytes.
    """
    import gzip
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return raw


def ticker_to_cik(ticker: str) -> str:
    """Resolve a ticker to a zero-padded 10-digit CIK via SEC's mapping file. RETURNS cik string."""
    data = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"ticker not found: {ticker}")


def latest_filing_doc(cik: str, form: str) -> tuple[str, str]:
    """Find the latest filing of `form` for a CIK. RETURNS (accession_no_dashes, primary_document).

    Reads the submissions JSON and scans the recent filings for the requested form type.
    """
    subs = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    recent = subs["filings"]["recent"]
    for i, ftype in enumerate(recent["form"]):
        if ftype == form:
            return recent["accessionNumber"][i].replace("-", ""), recent["primaryDocument"][i]
    raise ValueError(f"no {form} found for CIK {cik}")


def download(ticker: str, form: str, out_dir: str, to_pdf: bool = False) -> str:
    """Download one filing's primary document into out_dir. RETURNS the saved path.

    EDGAR serves primary docs as HTML. By default we save the .htm (the ingest pipeline extracts text
    from HTML directly — cleaner than PDF, which mangles tables). With to_pdf=True we additionally
    render it to PDF via pdfkit/wkhtmltopdf if that's installed.
    PARAM ticker : e.g. 'AAPL'.
    PARAM form   : e.g. '10-K'.
    PARAM out_dir: output directory.
    PARAM to_pdf : if True, also write a .pdf (requires `pip install pdfkit` + wkhtmltopdf binary).
    RETURNS: path to the saved file (.pdf if to_pdf and conversion succeeded, else .htm).
    """
    cik = ticker_to_cik(ticker)
    accession, doc = latest_filing_doc(cik, form)
    cik_int = str(int(cik))  # path uses un-padded CIK
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"
    content = _get(url)
    os.makedirs(out_dir, exist_ok=True)

    # Always save the HTML first; normalize the extension to .htm regardless of EDGAR's naming.
    base = os.path.splitext(os.path.basename(doc.replace("\\", "/")))[0]
    htm_path = os.path.join(out_dir, f"{ticker}_{form}_{base}.htm")
    with open(htm_path, "wb") as f:
        f.write(content)

    if not to_pdf:
        return htm_path

    # Optional HTML -> PDF. Needs pdfkit (pip) + the wkhtmltopdf binary on PATH.
    pdf_path = os.path.join(out_dir, f"{ticker}_{form}_{base}.pdf")
    try:
        import pdfkit
        pdfkit.from_file(htm_path, pdf_path)
        return pdf_path
    except Exception as e:
        print(f"   [pdf-skip] could not render PDF ({e}); keeping {os.path.basename(htm_path)}. "
              f"Install wkhtmltopdf + `pip install pdfkit` for PDF output. HTML works fine for ingest.")
        return htm_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--forms", nargs="+", default=DEFAULT_FORMS)
    ap.add_argument("--out", default="data")
    ap.add_argument("--pdf", action="store_true",
                    help="also render to PDF (requires wkhtmltopdf + pdfkit); default saves .htm")
    args = ap.parse_args()

    for ticker in args.tickers:
        for form in args.forms:
            try:
                path = download(ticker, form, args.out, to_pdf=args.pdf)
                print(f"[ok] {ticker} {form} -> {path}")
            except Exception as e:
                print(f"[skip] {ticker} {form}: {e}")
            time.sleep(0.5)  # be polite to EDGAR (rate limit)


if __name__ == "__main__":
    main()