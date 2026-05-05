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
  b.link AS download_link""",
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

# ── J. Ingredient → Diseases + Hadith + Reference ─────────────────────────

INGREDIENT_DISEASE_TREATMENT = CypherQuery(
        id="J",
        name="Ingredient → Disease → Hadith → Reference",
        description="Get diseases treated by ingredient + supporting hadith + reference",
        param_keys=("ingredient_name",),
        cypher="""\
MATCH (i:Ingredient)
WHERE toLower(i.name) = toLower($ingredient_name)
MATCH (i)-[:CURES]->(d:Disease)
OPTIONAL MATCH (d)-[:MENTIONED_IN]->(h:Hadith)
OPTIONAL MATCH (r:Reference)-[:HAS_HADITH]->(h)
RETURN
    i.name AS ingredient,
    d.name AS disease,
    h.name AS hadith_text,
    r.reference AS reference
""",
)


# ── K. Reverse Compound Lookup ──────────────────────────────────────────────

COMPOUND_INGREDIENTS = CypherQuery(
    id="K",
    name="Compound → Ingredients Reverse Lookup",
    description="Find ingredients/foods that contain a specific chemical compound",
    param_keys=("compound_name",),
    cypher="""\
MATCH (cc:ChemicalCompound)
WHERE toLower(cc.name) CONTAINS toLower($compound_name)
MATCH (i:Ingredient)-[r:CONTAINS]->(cc)
RETURN DISTINCT
  cc.name AS chemical_compound,
  i.name AS ingredient,
  r.quantity AS quantity,
  r.unit AS unit,
  r.food_part AS food_part,
  r.source AS source
ORDER BY ingredient
""",
)

# ── L. Ingredient → All Hadiths (Where Mentioned) ───────────────────────────

HADITH_WHERE_MENTIONED = CypherQuery(
    id="L",
    name="Ingredient → All Hadith Mentions",
    description="Get all hadiths where a food/ingredient is mentioned",
    param_keys=("food",),
    cypher="""\
MATCH (i:Ingredient)-[:MENTIONED_IN]->(h:Hadith)
WHERE toLower(i.name) CONTAINS toLower($food)
RETURN
  i.name AS Food,
  h.narrator AS Narrator,
  h.collection AS Collection,
  h.hadith_number AS HadithNumber,
  h.book AS Book,
  h.context_type AS ContextType,
  h.arabic_term AS ArabicTerm,
  h.disease_ref AS DiseaseRef,
  h.name AS HadithText
ORDER BY h.collection
""",
)

# ── M. Ingredient → Formal Citation ─────────────────────────────────────────

HADITH_FULL_CITATION = CypherQuery(
    id="M",
    name="Ingredient → Formal Hadith Citation",
    description="Get formal citation string for a food/ingredient across all hadiths",
    param_keys=("food",),
    cypher="""\
MATCH (i:Ingredient)-[:MENTIONED_IN]->(h:Hadith)
WHERE toLower(i.name) CONTAINS toLower($food)
RETURN
  i.name AS Food,
  h.collection + ' — ' + h.book + ', Hadith ' + h.hadith_number AS FormalCitation,
  h.narrator AS Narrator,
  h.context_type AS ContextType,
  h.disease_ref AS DiseaseRef,
  h.name AS FullHadithText
ORDER BY h.collection
""",
)

# ── N. Collection Filter (Bukhari / Muslim only) ─────────────────────────────

HADITH_COLLECTION_FILTER = CypherQuery(
    id="N",
    name="Ingredients in Sahih Bukhari or Sahih Muslim",
    description="List all ingredients mentioned in the two most authenticated collections",
    param_keys=(),
    cypher="""\
MATCH (i:Ingredient)-[:MENTIONED_IN]->(h:Hadith)
WHERE h.collection IN ['Sahih Bukhari', 'Sahih Muslim']
RETURN DISTINCT
  i.name AS Food,
  h.collection AS Collection,
  h.hadith_number AS HadithNumber,
  h.book AS Book,
  h.context_type AS ContextType,
  h.disease_ref AS DiseaseRef
ORDER BY h.collection, i.name
""",
)

# ── O. Context Type Filter (Medicinal / Dietary / Spiritual) ─────────────────

HADITH_CONTEXT_TYPE = CypherQuery(
    id="O",
    name="Ingredient → Hadith Context Type",
    description="Return hadiths for a food filtered by context type (medicinal/dietary/spiritual)",
    param_keys=("food",),
    cypher="""\
    MATCH (i:Ingredient)-[:MENTIONED_IN]->(h:Hadith)
    WHERE toLower(i.name) CONTAINS toLower($food)
    RETURN
    h.context_type AS ContextType,
    h.narrator AS Narrator,
    h.collection AS Collection,
    h.hadith_number AS HadithNumber,
    h.name AS HadithText
    ORDER BY h.context_type
    """,
)

# ── P. Arabic Term Lookup ────────────────────────────────────────────────────

HADITH_ARABIC_NAME = CypherQuery(
    id="P",
    name="Ingredient → Arabic Term via Hadith",
    description="Look up the Arabic name of a food as recorded in hadith sources",
    param_keys=("food",),
    cypher="""\
    MATCH (i:Ingredient)-[:MENTIONED_IN]->(h:Hadith)
    WHERE toLower(i.name) CONTAINS toLower($food)
    RETURN DISTINCT
    i.name AS Food,
    h.arabic_term AS ArabicName,
    h.collection + ' ' + h.hadith_number AS Citation,
    h.book AS Book,
    h.narrator AS Narrator
    ORDER BY h.collection
    """,
)

# ── Q. Mention Frequency per Collection ──────────────────────────────────────

HADITH_FREQUENCY = CypherQuery(
    id="Q",
    name="Ingredient → Mention Frequency by Collection",
    description="Count how many times a food is mentioned per hadith collection",
    param_keys=("food",),
    cypher="""\
    MATCH (i:Ingredient)-[:MENTIONED_IN]->(h:Hadith)
    WHERE toLower(i.name) CONTAINS toLower($food)
    RETURN
    h.collection AS Collection,
    count(h) AS MentionCount
    ORDER BY MentionCount DESC
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
        INGREDIENT_DISEASE_TREATMENT,
        COMPOUND_INGREDIENTS,
        
        # ── Obj-2: Hadith / Tibb-e-Nabawi ──
        HADITH_WHERE_MENTIONED,
        HADITH_FULL_CITATION,
        HADITH_COLLECTION_FILTER,
        HADITH_CONTEXT_TYPE,
        HADITH_ARABIC_NAME,
        HADITH_FREQUENCY,
    ]
}
