from pdfminer.high_level import extract_text
import os
import re

project_root = ".."
chakra_folder = os.path.join(project_root, "chakra_papers")
output_root = os.path.join(project_root, "chakra_split")
report_file = os.path.join(project_root, "results", "chakra_split_report.txt")

os.makedirs(output_root, exist_ok=True)
os.makedirs(os.path.join(project_root, "results"), exist_ok=True)

chakra_order = [
    ("root", [
        r"\broot chakra\b",
        r"\bmuladhara\b",
    ]),
    ("sacral", [
        r"\bsacral chakra\b",
        r"\bsvadhisthana\b",
        r"\bsvadisthana\b",
    ]),
    ("solar_plexus", [
        r"\bsolar plexus chakra\b",
        r"\bnavel chakra\b",
        r"\bmanipura\b",
    ]),
    ("heart", [
        r"\bheart chakra\b",
        r"\banahata\b",
    ]),
    ("throat", [
        r"\bthroat chakra\b",
        r"\bvishuddha\b",
        r"\bvisuddha\b",
    ]),
    ("third_eye", [
        r"\bthird eye chakra\b",
        r"\bajna\b",
    ]),
]

for chakra_name, _ in chakra_order:
    os.makedirs(os.path.join(output_root, chakra_name), exist_ok=True)

def normalize_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text

def find_first_match(text, patterns):
    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(m.start())
    return min(matches) if matches else None

report_lines = []

for filename in os.listdir(chakra_folder):
    if not filename.lower().endswith(".pdf"):
        continue

    path = os.path.join(chakra_folder, filename)
    print(f"Reading: {filename}")

    try:
        raw_text = extract_text(path)
    except Exception as e:
        report_lines.append(f"{filename}: FAILED to extract text ({e})")
        continue

    text = normalize_text(raw_text)

    positions = []
    for chakra_name, patterns in chakra_order:
        pos = find_first_match(text, patterns)
        if pos is not None:
            positions.append((chakra_name, pos))

    positions.sort(key=lambda x: x[1])

    if len(positions) < 3:
        report_lines.append(f"{filename}: WARNING - only found {len(positions)} chakra headings")
        continue

    found_names = [name for name, _ in positions]
    report_lines.append(f"{filename}: found headings -> {', '.join(found_names)}")

    for i, (chakra_name, start_pos) in enumerate(positions):
        end_pos = len(text)
        if i + 1 < len(positions):
            end_pos = positions[i + 1][1]

        section_text = text[start_pos:end_pos].strip()

        if len(section_text) < 100:
            report_lines.append(f"{filename}: WARNING - extracted very short section for {chakra_name}")
            continue

        base_name = os.path.splitext(filename)[0]
        out_name = f"{base_name}_{chakra_name}.txt"
        out_path = os.path.join(output_root, chakra_name, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(section_text)

with open(report_file, "w", encoding="utf-8") as f:
    for line in report_lines:
        f.write(line + "\n")

print("\nDone.")
print(f"Split files saved in: {output_root}")
print(f"Report saved in: {report_file}")
