"""
download_dataset.py —Downloads and cleans Wikipedia articles for training.
"""
import re

TARGET_SIZE_MB = 15

# Sections to cut — everything from these headers onwards is removed
CUT_SECTIONS = [
    "See also", "References", "External links", "Further reading",
    "Notes", "Citations", "Sources", "Bibliography",
    "Explanatory notes", "General and cited sources",
    "Primary sources", "Secondary sources", "Tertiary sources",
]


def clean_article(text):
    """Remove boilerplate sections and category tags from a Wikipedia article."""

    # Remove everything after any of the cut-section headers
    # Find the earliest matching boilerplate section and cut everything from there
    earliest_cut = len(text)
    for section in CUT_SECTIONS:
        # Match flexibly: start of line, optional whitespace, section name, optional whitespace
        pattern = r'\n\s*' + re.escape(section) + r'\s*\n'
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() < earliest_cut:
            earliest_cut = match.start()

    if earliest_cut < len(text):
        text = text[:earliest_cut]

    # Remove category tags at the bottom (lines that are just a single topic word)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty category-like lines (single words or very short lines at the end)
        # but keep them if they appear mid-article
        if stripped and len(stripped.split()) <= 3 and not any(c in stripped for c in '.,;:!?()'):
            # Could be a section header — keep it (they're useful context)
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Remove trailing whitespace and excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library not found. Install it with:")
        print("        pip install datasets")
        return

    output_file = "dataset.txt"
    target_bytes = TARGET_SIZE_MB * 1024 * 1024

    print("[INFO] Downloading Wikipedia articles from HuggingFace...")
    print(f"[INFO] Target size: ~{TARGET_SIZE_MB}MB (after cleaning)\n")

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    collected_bytes = 0
    doc_count = 0
    total_removed = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in ds:
            raw_text = sample.get("text", "")

            if not raw_text or len(raw_text.strip()) < 200:
                continue

            # Clean the article
            original_len = len(raw_text)
            cleaned = clean_article(raw_text)

            # Skip if article is too short after cleaning
            if len(cleaned) < 150:
                continue

            total_removed += (original_len - len(cleaned))

            if doc_count > 0:
                f.write("\n\n")
            f.write(cleaned)

            collected_bytes += len(cleaned.encode("utf-8"))
            doc_count += 1

            if doc_count % 500 == 0:
                mb = collected_bytes / (1024 * 1024)
                removed_mb = total_removed / (1024 * 1024)
                print(f"  Collected {doc_count} articles  ({mb:.1f} MB)  |  removed {removed_mb:.1f} MB of boilerplate")

            if collected_bytes >= target_bytes:
                break

    final_mb = collected_bytes / (1024 * 1024)
    removed_mb = total_removed / (1024 * 1024)
    print(f"\n[DONE] Saved {doc_count} cleaned Wikipedia articles ({final_mb:.1f} MB) to {output_file}")
    print(f"[INFO] Removed {removed_mb:.1f} MB of boilerplate (See also, References, External links, etc.)")
    print(f"[NEXT] Run:  python train.py")


if __name__ == "__main__":
    main()
