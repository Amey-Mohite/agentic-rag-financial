"""
download_filings.py — fetch a few public SEC filings into ./data for the demo corpus.

Uses SEC EDGAR's public endpoints. SEC requires a descriptive User-Agent with contact info;
set SEC_USER_AGENT in your environment (e.g. "Your Name your@email.com") to be a good citizen.

Usage:
    python scripts/download_filings.py
    python scripts/download_filings.py --tickers AAPL MSFT --forms 10-K 10-Q

FLOW: ticker → CIK (company id) → latest filing of a form type → download its primary document.
"""

from __future__ import annotations

import os
import argparse
import time            # for polite rate-limiting between requests
import json            # SEC endpoints return JSON
import urllib.request  # standard-library HTTP client (no extra dependency)


# A small default set chosen for VARIETY (dense 10-Ks), so retrieval has real work to do.
DEFAULT_TICKERS = ["AAPL", "MSFT"]
DEFAULT_FORMS = ["10-K"]

# SEC requires a descriptive User-Agent identifying who is making the request.
UA = os.environ.get("SEC_USER_AGENT", "agentic-rag-demo contact@example.com")
# Ask for an uncompressed ("identity") response. We still gunzip defensively below in case SEC
# compresses anyway.
HEADERS = {"User-Agent": UA, "Accept-Encoding": "identity"}


def _get(url: str) -> bytes:
    """HTTP GET with the SEC-required User-Agent, transparently decompressing gzip if present.

    WHY the gunzip guard: SEC sometimes returns gzip-compressed bodies (magic bytes 0x1f 0x8b)
    regardless of the request headers, and urllib does NOT auto-decompress. Without this,
    json.loads would receive raw gzip bytes and die with a UTF-8 decode error.

    Parameters
    ----------
    url : str
        The URL to fetch.

    Returns
    -------
    bytes
        The (decompressed) response body.
    """
    import gzip
    req = urllib.request.Request(url, headers=HEADERS)  # attach our headers to the request
    with urllib.request.urlopen(req) as r:
        raw = r.read()  # read the full response body
        # Decompress if the server flagged gzip OR the bytes start with the gzip magic number.
        if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return raw


def ticker_to_cik(ticker: str) -> str:
    """Resolve a stock ticker (e.g. 'AAPL') to its zero-padded 10-digit SEC CIK.

    The CIK (Central Index Key) is SEC's internal company identifier; every EDGAR endpoint is
    keyed by it. SEC publishes a ticker→CIK mapping file we scan here.

    Returns
    -------
    str
        The 10-digit, zero-padded CIK (e.g. "0000320193").
    """
    data = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    # The file is a dict of rows; find the one whose ticker matches (case-insensitive).
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)  # zero-pad to 10 digits as EDGAR expects
    raise ValueError(f"ticker not found: {ticker}")


def latest_filing_doc(cik: str, form: str) -> tuple[str, str]:
    """Find the most recent filing of a given form type for a company.

    Reads the company's submissions JSON and scans its recent filings for the first matching
    form (the list is newest-first, so the first match is the latest).

    Returns
    -------
    tuple[str, str]
        (accession_number_without_dashes, primary_document_filename).
    """
    subs = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    recent = subs["filings"]["recent"]  # parallel arrays: form[i], accessionNumber[i], etc.
    for i, ftype in enumerate(recent["form"]):
        if ftype == form:
            # Strip dashes from the accession number (the URL path uses the dashless form).
            return recent["accessionNumber"][i].replace("-", ""), recent["primaryDocument"][i]
    raise ValueError(f"no {form} found for CIK {cik}")


def download(ticker: str, form: str, out_dir: str, to_pdf: bool = False) -> str:
    """Download one filing's primary document into out_dir and return the saved path.

    EDGAR serves primary docs as HTML. By default we save the .htm (the ingest pipeline extracts
    text from HTML directly — cleaner than PDF, which mangles tables). With to_pdf=True we
    additionally render it to PDF via pdfkit/wkhtmltopdf if that toolchain is installed.

    Parameters
    ----------
    ticker : str
        e.g. 'AAPL'.
    form : str
        e.g. '10-K'.
    out_dir : str
        Output directory (created if missing).
    to_pdf : bool
        If True, also write a .pdf (requires `pip install pdfkit` + the wkhtmltopdf binary).

    Returns
    -------
    str
        Path to the saved file (.pdf if requested and conversion succeeded, else .htm).
    """
    cik = ticker_to_cik(ticker)                 # ticker → CIK
    accession, doc = latest_filing_doc(cik, form)  # CIK + form → (accession, primary doc)
    cik_int = str(int(cik))                     # the archive path uses the UN-padded CIK
    # Build the archive URL for the primary document and download it.
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"
    content = _get(url)
    os.makedirs(out_dir, exist_ok=True)         # ensure the output directory exists

    # Always save the HTML first; normalize the extension to .htm regardless of EDGAR's naming.
    base = os.path.splitext(os.path.basename(doc.replace("\\", "/")))[0]  # filename without extension
    htm_path = os.path.join(out_dir, f"{ticker}_{form}_{base}.htm")
    with open(htm_path, "wb") as f:             # "wb": EDGAR content is bytes
        f.write(content)

    if not to_pdf:
        return htm_path  # default: HTML is all we need for ingest

    # --- Optional HTML → PDF. Needs pdfkit (pip) + the wkhtmltopdf binary on PATH.
    pdf_path = os.path.join(out_dir, f"{ticker}_{form}_{base}.pdf")
    try:
        import pdfkit
        pdfkit.from_file(htm_path, pdf_path)
        return pdf_path
    except Exception as e:
        # Conversion is best-effort: if the toolchain is missing, keep the HTML and explain.
        print(f"   [pdf-skip] could not render PDF ({e}); keeping {os.path.basename(htm_path)}. "
              f"Install wkhtmltopdf + `pip install pdfkit` for PDF output. HTML works fine for ingest.")
        return htm_path


def main():
    """Parse args and download each (ticker, form) combination, being polite to EDGAR."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)  # one or more tickers
    ap.add_argument("--forms", nargs="+", default=DEFAULT_FORMS)      # one or more form types
    ap.add_argument("--out", default="data")                         # output directory
    ap.add_argument("--pdf", action="store_true",
                    help="also render to PDF (requires wkhtmltopdf + pdfkit); default saves .htm")
    args = ap.parse_args()

    # Nested loop: download every form for every ticker.
    for ticker in args.tickers:
        for form in args.forms:
            try:
                path = download(ticker, form, args.out, to_pdf=args.pdf)
                print(f"[ok] {ticker} {form} -> {path}")
            except Exception as e:
                # One failure shouldn't abort the whole batch — log and continue.
                print(f"[skip] {ticker} {form}: {e}")
            time.sleep(0.5)  # be polite to EDGAR (simple rate limit between requests)


if __name__ == "__main__":
    main()
