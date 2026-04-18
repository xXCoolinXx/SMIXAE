import csv
import re

import pdfplumber


def convert_anew_to_csv(pdf_path, output_csv):
    # Table 1 (All Subjects) resides on pages 5 through 18 (indices 4 to 17)
    start_page = 4
    end_page = 17

    # Regex to capture: Word, Valence Mean, and Arousal Mean
    # Accounts for multi-column layout and standard deviation parentheses
    row_pattern = re.compile(r'([a-zA-Z]+)\s+(?:\d+\s+)?(\d\.\d{2})\s+\(\d\.\d{2}\)\s+(\d\.\d{2})\s+\(\d\.\d{2}\)')

    with pdfplumber.open(pdf_path) as pdf:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Word', 'Valence', 'Arousal'])

            buffer = []

            for i in range(start_page, end_page + 1):
                page = pdf.pages[i]
                text = page.extract_text()

                if text:
                    for match in row_pattern.finditer(text):
                        word = match.group(1).lower()
                        valence = match.group(2)
                        arousal = match.group(3)

                        buffer.append([word, valence, arousal])

            buffer.sort()
            for row in buffer:
                writer.writerow(row)

convert_anew_to_csv('anew.pdf', 'anew_scores.csv')
