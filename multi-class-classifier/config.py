# config.py

HEADS = [
    ("A_adaptive", 8),
    ("A_maladaptive", 8),
    ("B-O_adaptive", 3),
    ("B-O_maladaptive", 3),
    ("B-S_adaptive", 2),
    ("B-S_maladaptive", 2),
    ("C-O_adaptive", 3),
    ("C-O_maladaptive", 3),
    ("C-S_adaptive", 2),
    ("C-S_maladaptive", 2),
    ("D_adaptive", 4),
    ("D_maladaptive", 4),
]
TEXT_COLUMN = "post_text"
CSV_FILE = "transformer_preprocessed.csv"