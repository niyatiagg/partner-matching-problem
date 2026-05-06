"""Generate teammate CSVs. Run: python scripts/generate_teammate_csvs.py"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STUDY_INTERESTS_CS = [
    "AI-ML",
    "Cybersecurity",
    "Blockchain",
    "Software Engineering",
    "Data Science & Data Mining",
    "Computer Architecture & Hardware",
    "Networking & Distributed Systems",
    "Computer Graphics & Vizualization",
    "Human-Computer Interaction",
    "Robotics",
    "Theory of Computation",
    "Cloud Computing",
    "Database Systems",
]

PURPOSES_CS = [
    "good grades- high time investment",
    "decent grades- medium time investment",
    "passable grades- low time investment",
    "research",
    "hackathon focused",
    "startup idea",
]

# Hogwarts: same as CS teammate purposes except hackathon.
PURPOSES_HOGWARTS = [p for p in PURPOSES_CS if p != "hackathon focused"]

MEETING = ["online", "offline", "both ok"]
LANGS = ["python", "java", "c++"]
DOMAIN = ["frontend", "backend"]

WORK_PREF_HOGWARTS = [
    "Researcher/ Theoretical work",
    "Explorer/ Practical Implementation",
]

HOGWARTS_STUDY = [
    "Transfiguration",
    "Charms",
    "Potions",
    "Defense against the Dark Arts",
    "Herbology",
    "Study of Ancient Runes",
    "Divination",
    "Muggle Studies",
    "Apparition",
    "Arithmancy",
    "Care of Magical Creatures",
]

FIRST = [
    "Aarav",
    "Isha",
    "Neha",
    "Rohan",
    "Diya",
    "Vikram",
    "Kabir",
    "Aditi",
    "Meera",
    "Arjun",
    "Zara",
    "Dev",
    "Anaya",
    "Kiran",
    "Saanvi",
    "Reyansh",
    "Tara",
    "Vihaan",
    "Myra",
    "Shaurya",
]

# 100 canonical Harry Potter universe names (books + films; minor characters included).
HP_CHARACTER_NAMES = [
    "Harry Potter",
    "Ronald Weasley",
    "Hermione Granger",
    "Ginny Weasley",
    "Neville Longbottom",
    "Dean Thomas",
    "Parvati Patil",
    "Padma Patil",
    "Draco Malfoy",
    "Vincent Crabbe",
    "Gregory Goyle",
    "Pansy Parkinson",
    "Blaise Zabini",
    "Theodore Nott",
    "Luna Lovegood",
    "Cho Chang",
    "Cedric Diggory",
    "Viktor Krum",
    "Fleur Delacour",
    "Oliver Wood",
    "Angelina Johnson",
    "Alicia Spinnet",
    "Katie Bell",
    "Fred Weasley",
    "George Weasley",
    "Percy Weasley",
    "Charlie Weasley",
    "Bill Weasley",
    "Arthur Weasley",
    "Molly Weasley",
    "Remus Lupin",
    "Sirius Black",
    "Severus Snape",
    "Minerva McGonagall",
    "Albus Dumbledore",
    "Rubeus Hagrid",
    "Horace Slughorn",
    "Pomona Sprout",
    "Filius Flitwick",
    "Sybill Trelawney",
    "Poppy Pomfrey",
    "Gilderoy Lockhart",
    "Quirinus Quirrell",
    "Argus Filch",
    "Rolanda Hooch",
    "Septima Vector",
    "Aurora Sinistra",
    "Alastor Moody",
    "Kingsley Shacklebolt",
    "Nymphadora Tonks",
    "Bellatrix Lestrange",
    "Lucius Malfoy",
    "Narcissa Malfoy",
    "Peter Pettigrew",
    "James Potter",
    "Lily Potter",
    "Petunia Dursley",
    "Vernon Dursley",
    "Dudley Dursley",
    "Aberforth Dumbledore",
    "Gellert Grindelwald",
    "Rita Skeeter",
    "Dolores Umbridge",
    "Cornelius Fudge",
    "Rufus Scrimgeour",
    "Barty Crouch Jr",
    "Barty Crouch Sr",
    "Igor Karkaroff",
    "Olympe Maxime",
    "Marcus Flint",
    "Terence Higgs",
    "Adrian Pucey",
    "Hannah Abbott",
    "Susan Bones",
    "Ernie Macmillan",
    "Justin Finch-Fletchley",
    "Zacharias Smith",
    "Marietta Edgecombe",
    "Roger Davies",
    "Michael Corner",
    "Anthony Goldstein",
    "Terry Boot",
    "Dennis Creevey",
    "Colin Creevey",
    "Seamus Finnigan",
    "Lavender Brown",
    "Romilda Vane",
    "Cormac McLaggen",
    "Lee Jordan",
    "Stewart Ackerley",
    "Orla Quirke",
    "Lisa Turpin",
    "Sally-Anne Perks",
    "Morag MacDougal",
    "Wayne Hopkins",
    "Megan Jones",
    "Kevin Entwhistle",
    "Lily Moon",
    "Oliver Rivers",
    "Sally Smith",
    "Jack Sloper",
    "Andrew Kirke",
    "Jimmy Peakes",
    "Ritchie Coote",
    "Demelza Robins",
    "Euan Abercrombie",
    "Rose Zeller",
]

assert len(HP_CHARACTER_NAMES) >= 100


def make_cs_rows(region_label: str, seed: int, n: int = 100) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        uid = f"{region_label[:4].upper().replace(' ', '')}_{i+1:03d}"
        name = f"{rng.choice(FIRST)} {rng.choice(FIRST)}"
        rows.append(
            {
                "user_id": uid,
                "name": name,
                "purpose": rng.choice(PURPOSES_CS),
                "meeting_preference": rng.choice(MEETING),
                "preferred_language": rng.choice(LANGS),
                "study_interests": rng.choice(STUDY_INTERESTS_CS),
                "domain": rng.choice(DOMAIN),
                "openness": rng.randint(1, 10),
                "conscientiousness": rng.randint(1, 10),
                "extraversion": rng.randint(1, 10),
                "agreeableness": rng.randint(1, 10),
                "neuroticism": rng.randint(1, 10),
                "region": region_label,
            }
        )
    return pd.DataFrame(rows)


def make_hogwarts_rows(region_label: str, seed: int, n: int = 100) -> pd.DataFrame:
    rng = random.Random(seed)
    # Fixed order: main cast first (matches ~100-name list length).
    names = list(HP_CHARACTER_NAMES[:n])
    rows = []
    for i in range(n):
        rows.append(
            {
                "user_id": f"HOGW_{i+1:03d}",
                "name": names[i],
                "purpose": rng.choice(PURPOSES_HOGWARTS),
                "meeting_preference": rng.choice(MEETING),
                "study_interests": rng.choice(HOGWARTS_STUDY),
                "work_preference": rng.choice(WORK_PREF_HOGWARTS),
                "openness": rng.randint(1, 10),
                "conscientiousness": rng.randint(1, 10),
                "extraversion": rng.randint(1, 10),
                "agreeableness": rng.randint(1, 10),
                "neuroticism": rng.randint(1, 10),
                "region": region_label,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cs = make_cs_rows("Computer Science - Class 2026", seed=42)
    hw = make_hogwarts_rows("Hogwarts - Class 2026", seed=43)
    cs.to_csv(ROOT / "computer_science_class_2026.csv", index=False)
    hw.to_csv(ROOT / "hogwarts_class_2026.csv", index=False)
    (ROOT / "data" / "regions").mkdir(parents=True, exist_ok=True)
    cs.to_csv(ROOT / "data" / "regions" / "computer_science_class_2026.csv", index=False)
    hw.to_csv(ROOT / "data" / "regions" / "hogwarts_class_2026.csv", index=False)
    print("Wrote computer_science_class_2026.csv and hogwarts_class_2026.csv (100 rows each).")


if __name__ == "__main__":
    main()
