"""CLI interface using Click."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import click

from . import log as _log
from .catalog import Catalog, Record, load_csv
from .extract import ExtractOptions, Extractor, Details
from .imageutil import preprocess_and_save
from .ocr import create_ocr_client, init_worker_client, get_worker_client
from .reference import enrich_record
from .sync import sync_reference_file


@click.group()
def main():
    """Pokemon TCG card detection and cataloging."""
    pass


@main.command()
@click.option("--image", required=True, help="Caminho da imagem da carta")
@click.option("--csv", "csv_path", required=True, help="Caminho do CSV")
@click.option("--quality", default="M", help="Qualidade padrao")
@click.option("--language", default="PT", help="Idioma padrao")
@click.option("--quantity", default=1, type=int, help="Quantidade padrao")
@click.option("--write", "do_write", is_flag=True, help="Gravar no CSV")
@click.option("--ocr-backend", default=None, help="Backend OCR: easyocr ou tesseract")
@click.option("--gpu/--no-gpu", default=False, help="Usar GPU para EasyOCR")
def extract(image: str, csv_path: str, quality: str, language: str, quantity: int, do_write: bool, ocr_backend: str | None, gpu: bool):
    """Extract card data from a single image."""
    cat = _load_or_create_catalog(csv_path)
    client = create_ocr_client(backend=ocr_backend, gpu=gpu)
    extractor = Extractor(client, ExtractOptions(quality, language, quantity))

    record, details = extractor.extract(image)
    ocr_name = details.raw_text.get("nome_alternativo", "") or details.raw_text.get("nome", "")
    from .parsing import sanitize_name as _sn
    enrich_record(cat, record, ocr_name=_sn(ocr_name))
    _print_summary(record, details)

    if do_write:
        errors = record.validate_for_write()
        if errors:
            click.echo(f"Validacao falhou: {'; '.join(errors)}", err=True)
            sys.exit(1)
        if cat.exists(record.edition_sigla, record.card_number):
            click.echo(f"Registro duplicado: {record.edition_sigla} #{record.card_number}", err=True)
            sys.exit(1)
        cat.append(record, flush=True)
        click.echo("linha adicionada ao CSV")


@main.command()
@click.option("--images-file", required=True, help="Arquivo com caminhos de imagens")
@click.option("--csv", "csv_path", required=True, help="Caminho do CSV")
@click.option("--quality", default="M")
@click.option("--language", default="PT")
@click.option("--quantity", default=1, type=int)
@click.option("--write", "do_write", is_flag=True)
@click.option("--error-log", default="batch-errors.jsonl")
@click.option("--log-file", default="pokemon-tcg.log", help="Arquivo de log ('' para desativar)")
@click.option("--ocr-backend", default=None)
@click.option("--gpu/--no-gpu", default=False)
@click.option("--workers", default=1, type=int, help="Numero de workers paralelos")
def batch(images_file: str, csv_path: str, quality: str, language: str, quantity: int, do_write: bool, error_log: str, log_file: str, ocr_backend: str | None, gpu: bool, workers: int):
    """Process multiple images in batch."""
    _log.setup(log_file or None)
    cat = _load_or_create_catalog(csv_path)
    image_paths = _read_image_list(images_file)
    if not image_paths:
        click.echo("Nenhuma imagem para processar", err=True)
        sys.exit(1)

    workers = max(1, min(workers, len(image_paths)))

    _logger = _log.get("cli")
    _logger.info(
        "[BATCH] inicio: %d imagens, csv=%s, workers=%d, write=%s",
        len(image_paths), csv_path, workers, do_write,
    )
    t_batch = time.perf_counter()

    if workers > 1:
        _batch_parallel(cat, image_paths, csv_path, quality, language, quantity, do_write, error_log, ocr_backend, gpu, workers)
    else:
        _batch_sequential(cat, image_paths, quality, language, quantity, do_write, error_log, ocr_backend, gpu)

    _logger.info("[BATCH] fim: %.3fs total", time.perf_counter() - t_batch)


def _batch_sequential(cat: Catalog, image_paths: list[str], quality: str, language: str, quantity: int, do_write: bool, error_log: str, ocr_backend: str | None, gpu: bool):
    """Process images one at a time (original flow)."""
    _logger = _log.get("cli")
    client = create_ocr_client(backend=ocr_backend, gpu=gpu)
    extractor = Extractor(client, ExtractOptions(quality, language, quantity))

    processed = succeeded = failed = duplicates = 0
    error_file = open(error_log, "a", encoding="utf-8") if error_log else None

    try:
        for img_path in image_paths:
            processed += 1
            t_img = time.perf_counter()
            try:
                record, details = extractor.extract(img_path)
                ocr_name = details.raw_text.get("nome_alternativo", "") or details.raw_text.get("nome", "")
                from .parsing import sanitize_name as _sn
                enrich_record(cat, record, ocr_name=_sn(ocr_name))
                errors = record.validate_for_write()
                if errors:
                    raise ValueError(f"extracao automatica incompleta: {'; '.join(errors)}")
                if do_write:
                    if cat.exists(record.edition_sigla, record.card_number):
                        duplicates += 1
                        _logger.info("[SKIP] duplicado: %s -> %s #%s", img_path, record.card_pt, record.card_number)
                        click.echo(f"  SKIP (duplicado) {img_path}")
                        continue
                    cat.append(record)
                    _logger.info("[CSV] gravando: %s #%s (%s) <- %s", record.card_pt, record.card_number, record.edition_sigla, img_path)
                succeeded += 1
                _logger.info("[OK] %.3fs — %s -> %s #%s", time.perf_counter() - t_img, img_path, record.card_pt, record.card_number)
                click.echo(f"  OK   {img_path} -> {record.card_pt} #{record.card_number}")
            except Exception as e:
                failed += 1
                _logger.error("[FAIL] %.3fs — %s: %s", time.perf_counter() - t_img, img_path, e)
                click.echo(f"  FAIL {img_path}: {e}", err=True)
                if error_file:
                    entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "imagePath": img_path,
                        "error": str(e),
                    }
                    error_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            finally:
                gc.collect()
    finally:
        cat.flush()
        if error_file:
            error_file.close()

    click.echo(f"\nResumo: {processed} processadas, {succeeded} ok, {failed} falhas, {duplicates} duplicadas")
    _logger.info("[BATCH] resumo: %d processadas, %d ok, %d falhas, %d duplicadas", processed, succeeded, failed, duplicates)
    if failed > 0:
        sys.exit(1)


def _extract_single(img_path: str, quality: str, language: str, quantity: int) -> tuple[str, Record | None, str | None]:
    """Worker function for parallel batch. Runs in a subprocess.

    Returns only (img_path, record, ocr_name_or_error) — intentionally omits
    Details to avoid serialising large OCR raw-text dicts across processes.
    """
    try:
        client = get_worker_client()
        extractor = Extractor(client, ExtractOptions(quality, language, quantity))
        record, details = extractor.extract(img_path)
        ocr_name = details.raw_text.get("nome_alternativo", "") or details.raw_text.get("nome", "")
        from .parsing import sanitize_name as _sn
        return (img_path, record, _sn(ocr_name))
    except Exception as e:
        return (img_path, None, str(e))


def _batch_parallel(cat: Catalog, image_paths: list[str], csv_path: str, quality: str, language: str, quantity: int, do_write: bool, error_log: str, ocr_backend: str | None, gpu: bool, workers: int):
    """Process images in parallel using multiple processes."""
    processed = succeeded = failed = duplicates = 0
    error_file = open(error_log, "a", encoding="utf-8") if error_log else None

    click.echo(f"Processando {len(image_paths)} imagens com {workers} workers...")

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker_client,
            initargs=(ocr_backend, gpu),
        ) as executor:
            futures = {
                executor.submit(_extract_single, img, quality, language, quantity): img
                for img in image_paths
            }
            # Process each result as soon as it arrives — avoids accumulating all
            # Record objects and OCR data in memory before any are written.
            for future in as_completed(futures):
                processed += 1
                img_path = futures[future]
                path, record, extra = future.result()
                if record is None:
                    # extra contains the error message
                    failed += 1
                    click.echo(f"  FAIL {img_path}: {extra}", err=True)
                    if error_file:
                        entry = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "imagePath": img_path,
                            "error": extra,
                        }
                        error_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    continue

                ocr_name = extra  # sanitized name from worker
                enrich_record(cat, record, ocr_name=ocr_name)
                errors = record.validate_for_write()
                if errors:
                    failed += 1
                    click.echo(f"  FAIL {img_path}: extracao automatica incompleta: {'; '.join(errors)}", err=True)
                    if error_file:
                        entry = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "imagePath": img_path,
                            "error": f"extracao automatica incompleta: {'; '.join(errors)}",
                        }
                        error_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    continue

                if do_write:
                    if cat.exists(record.edition_sigla, record.card_number):
                        duplicates += 1
                        click.echo(f"  SKIP (duplicado) {img_path}")
                        continue
                    cat.append(record)
                succeeded += 1
                click.echo(f"  OK   {img_path} -> {record.card_pt} #{record.card_number}")
    finally:
        cat.flush()
        if error_file:
            error_file.close()

    click.echo(f"\nResumo: {processed} processadas, {succeeded} ok, {failed} falhas, {duplicates} duplicadas")
    if failed > 0:
        sys.exit(1)


@main.command()
@click.option("--input-dir", required=True, help="Pasta com imagens")
@click.option("--output-dir", required=True, help="Pasta de saida")
@click.option("--contrast", default=35.0, type=float)
@click.option("--overwrite", is_flag=True)
def preprocess(input_dir: str, output_dir: str, contrast: float, overwrite: bool):
    """Preprocess card images (grayscale + contrast)."""
    import glob

    patterns = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(Path(input_dir) / pat)))
    files.sort()

    if not files:
        click.echo("Nenhuma imagem encontrada")
        return

    for f in files:
        out_path = str(Path(output_dir) / (Path(f).stem + ".png"))
        if not overwrite and Path(out_path).exists():
            click.echo(f"  SKIP {f}")
            continue
        try:
            result = preprocess_and_save(f, out_path, contrast)
            click.echo(f"  OK   {f} -> {out_path} (card_found={result['card_found']})")
        except Exception as e:
            click.echo(f"  FAIL {f}: {e}", err=True)


@main.command("sync")
@click.option("--source", default="pokemontcg", type=click.Choice(["pokemontcg", "tcgdex"]))
@click.option("--query", default="")
@click.option("--edition-ptbr", default="")
@click.option("--edition-sigla", default="")
@click.option("--out", "output_path", default="reference/reference_cards.json")
@click.option("--overrides", default="reference/reference_overrides.json")
@click.option("--language", default="PT")
@click.option("--quality", default="M")
@click.option("--quantity", default=1, type=int)
@click.option("--tcgdex-set-path", default="")
@click.option("--tcgdex-locale", default="pt")
def sync_cmd(source: str, query: str, edition_ptbr: str, edition_sigla: str, output_path: str, overrides: str, language: str, quality: str, quantity: int, tcgdex_set_path: str, tcgdex_locale: str):
    """Sync reference data from external APIs."""
    try:
        sync_reference_file(
            source=source,
            query=query,
            edition_ptbr=edition_ptbr,
            edition_sigla=edition_sigla,
            language=language,
            quality=quality,
            quantity=quantity,
            output_path=output_path,
            overrides_path=overrides,
            tcgdex_set_path=tcgdex_set_path,
            tcgdex_locale=tcgdex_locale,
        )
        click.echo(f"Referencia sincronizada em {output_path}")
    except Exception as e:
        click.echo(f"Erro: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_or_create_catalog(path: str) -> Catalog:
    if Path(path).exists():
        return load_csv(path)
    from .catalog import HEADER

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        escaped = ['"' + h.replace('"', '""') + '"' for h in HEADER]
        f.write(",".join(escaped) + "\n")
    return Catalog(path, [])


def _read_image_list(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _print_summary(record: Record, details: Details):
    click.echo(f"Card (PT):       {record.card_pt}")
    click.echo(f"Card (EN):       {record.card_en}")
    click.echo(f"Edicao (PTBR):   {record.edition_ptbr}")
    click.echo(f"Edicao (EN):     {record.edition_en}")
    click.echo(f"Edicao (Sigla):  {record.edition_sigla}")
    click.echo(f"Card #:          {record.card_number}")
    click.echo(f"# Cards:         {record.edition_card_count}")
    click.echo(f"Raridade:        {record.rarity}")
    click.echo(f"Cor:             {record.color}")
    click.echo(f"Quantidade:      {record.quantity}")
    click.echo(f"Qualidade:       {record.quality}")
    click.echo(f"Idioma:          {record.language}")
    if details.notes:
        click.echo(f"Notas:           {'; '.join(details.notes)}")
    if details.raw_text:
        click.echo("--- Raw OCR ---")
        for region, text in details.raw_text.items():
            click.echo(f"  [{region}]: {text.strip()[:80]}")


if __name__ == "__main__":
    main()
