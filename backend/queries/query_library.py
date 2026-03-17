"""Predefined Cypher Query Library.

Every query in this module is lifted verbatim from ``temp.md``.
NO dynamic Cypher generation is permitted anywhere in the codebase.

Each constant is a tuple of ``(cypher_template, param_keys)`` so that the
graph_service can validate that the caller supplies the right parameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CypherQuery:
    """Immutable container for a predefined Cypher query."""

    id: str
    name: str
    description: str
    cypher: str
    param_keys: tuple[str, ...]


# ── A. Disease → Ingredient → Hadith → Reference ────────────────────────────

DISEASE_INGREDIENT_HADITH = CypherQuery(
    id="A",
    name="Disease → Ingredient → Hadith → Reference",
    description="Get ingredients that cure disease + supporting hadith + reference",
    param_keys=("disease_name",),
    cypher="""\
MATCH (d:Disease)
WHERE toLower(d.name) = toLower($disease_name)
MATCH (i:Ingredient)-[:CURES]->(d)
OPTIONAL MATCH (d)-[:MENTIONED_IN]->(h:Hadith)
OPTIONAL MATCH (r:Reference)-[:HAS_HADITH]->(h)
RETURN
  i.name AS ingredient,
  h.name AS hadith_text,
  r.reference AS reference
LIMIT 20
""",
)

# ── B. Ingredient → Compound (with source) ──────────────────────────────────

INGREDIENT_COMPOUNDS = CypherQuery(
    id="B",
    name="Ingredient → Compounds WITH Source + Metadata",
    description="Get compounds + source + quantity for an ingredient",
    param_keys=("ingredient_name",),
    cypher="""\
MATCH (i:Ingredient)
WHERE toLower(i.name) = toLower($ingredient_name)
MATCH (i)-[r:CONTAINS]->(c:ChemicalCompound)
RETURN
  c.name AS compound,
  r.source AS source,
  r.quantity AS quantity,
  r.unit AS unit,
  r.food_part AS food_part
LIMIT 50
""",
)

# ── C. Ingredient → Compound → DrugCompound → Drug (full trace) ─────────────

INGREDIENT_DRUG_MAPPING = CypherQuery(
    id="C",
    name="Ingredient → Drug Mapping WITH SOURCE TRACE",
    description="Full mapping chain including source of drug relation",
    param_keys=("ingredient_name",),
    cypher="""\
MATCH (i:Ingredient)
WHERE toLower(i.name) = toLower($ingredient_name)
MATCH (i)-[r1:CONTAINS]->(c:ChemicalCompound)
MATCH (c)-[r2:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc:DrugChemicalCompound)
MATCH (d:Drug)-[r3:CONTAINS]->(dcc)
RETURN
  d.name AS drug,
  c.name AS compound,
  type(r2) AS mapping_strength,
  r1.source AS ingredient_source,
  r3.source AS drug_source
LIMIT 30
""",
)

# ── D1. Drug → Book → Download Link ─────────────────────────────────────────

DRUG_BOOK_LINK = CypherQuery(
    id="D1",
    name="Drug Source → Book Download Link",
    description="Resolve drug → book → download link",
    param_keys=("drug_name",),
    cypher="""\
MATCH (d:Drug)
WHERE toLower(d.name) = toLower($drug_name)
MATCH (d)-[:IS_IN_BOOK]->(b:Book)
RETURN
  b.name AS book_name,
  b.link AS download_link
LIMIT 5
""",
)

# ── D2. Drug via Source → Book (indirect) ────────────────────────────────────

DRUG_BOOK_INDIRECT = CypherQuery(
    id="D2",
    name="Drug via Source → Book (Indirect)",
    description="Match book name stored in relationship.source",
    param_keys=("ingredient_name",),
    cypher="""\
MATCH (i:Ingredient)
WHERE toLower(i.name) = toLower($ingredient_name)
MATCH (i)-[:CONTAINS]->(c:ChemicalCompound)
MATCH (c)-[:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc:DrugChemicalCompound)
MATCH (d:Drug)-[r:CONTAINS]->(dcc)
MATCH (b:Book)
WHERE toLower(b.name) = toLower(r.source)
RETURN DISTINCT b.name AS book_name, b.link AS link
LIMIT 10
""",
)

# ── E. Disease → Full Explainable Chain ──────────────────────────────────────

DISEASE_FULL_CHAIN = CypherQuery(
    id="E",
    name="Disease → Ingredient → Compound → Drug → Source → Book",
    description="Complete explainable pipeline from disease to book",
    param_keys=("disease_name",),
    cypher="""\
MATCH (d:Disease)
WHERE toLower(d.name) = toLower($disease_name)
MATCH (i:Ingredient)-[:CURES]->(d)
MATCH (i)-[r1:CONTAINS]->(c:ChemicalCompound)
MATCH (c)-[r2:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc:DrugChemicalCompound)
MATCH (dr:Drug)-[r3:CONTAINS]->(dcc)
OPTIONAL MATCH (b:Book)
WHERE toLower(b.name) = toLower(r3.source)
RETURN
  i.name AS ingredient,
  c.name AS compound,
  dr.name AS drug,
  type(r2) AS strength,
  r1.source AS ingredient_source,
  r3.source AS drug_source,
  b.link AS book_link
LIMIT 25
""",
)

# ── F. Hadith Deep Linking ───────────────────────────────────────────────────

HADITH_DEEP_LINK = CypherQuery(
    id="F",
    name="Hadith → Reference → Book",
    description="Full hadith source chain for a disease",
    param_keys=("disease_name",),
    cypher="""\
MATCH (d:Disease)
WHERE toLower(d.name) = toLower($disease_name)
MATCH (d)-[:MENTIONED_IN]->(h:Hadith)
MATCH (r:Reference)-[:HAS_HADITH]->(h)
OPTIONAL MATCH (b:Book)
WHERE toLower(b.name) = toLower(r.reference)
RETURN
  h.name AS hadith,
  r.reference AS reference,
  b.link AS book_link
LIMIT 10
""",
)

# ── G. Source Validation ─────────────────────────────────────────────────────

SOURCE_VALIDATION = CypherQuery(
    id="G",
    name="Check if Source Exists as Book",
    description="Validate whether a source string matches an existing Book node",
    param_keys=("source_name",),
    cypher="""\
MATCH (b:Book)
WHERE toLower(b.name) = toLower($source_name)
RETURN count(b) > 0 AS exists
""",
)

# ── H. Advanced Filtering ───────────────────────────────────────────────────

ADVANCED_FILTERING = CypherQuery(
    id="H",
    name="Exclude Weak + Keep Provenance",
    description="List compound → drug-compound mappings with relationship types",
    param_keys=(),
    cypher="""\
MATCH (c:ChemicalCompound)-[r:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc)
RETURN c.name, dcc.name, type(r)
LIMIT 50
""",
)

# ── I. Count Drugs per Ingredient ────────────────────────────────────────────

COUNT_DRUGS_FOR_INGREDIENT = CypherQuery(
    id="I",
    name="Count Drugs per Ingredient",
    description="Count distinct drugs mapped through identical/equivalent compounds",
    param_keys=("ingredient_name",),
    cypher="""\
MATCH (i:Ingredient)
WHERE toLower(i.name) = toLower($ingredient_name)
MATCH (i)-[:CONTAINS]->(c)
MATCH (c)-[:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc)
MATCH (d:Drug)-[:CONTAINS]->(dcc)
RETURN count(DISTINCT d) AS drug_count
""",
)


# ── Lookup by ID ─────────────────────────────────────────────────────────────

ALL_QUERIES: dict[str, CypherQuery] = {
    q.id: q
    for q in [
        DISEASE_INGREDIENT_HADITH,
        INGREDIENT_COMPOUNDS,
        INGREDIENT_DRUG_MAPPING,
        DRUG_BOOK_LINK,
        DRUG_BOOK_INDIRECT,
        DISEASE_FULL_CHAIN,
        HADITH_DEEP_LINK,
        SOURCE_VALIDATION,
        ADVANCED_FILTERING,
        COUNT_DRUGS_FOR_INGREDIENT,
    ]
}
