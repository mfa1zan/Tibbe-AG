## A. Disease → Ingredient → Hadith → Reference (FULL TRACE)

**USE_CASE_NAME:**

Disease → Ingredient → Hadith → Reference (Traceable Cure Evidence)

**DESCRIPTION:**
Get ingredients that cure disease + supporting hadith + reference

**INPUTS:**
disease_name

**CYPHER_QUERY:**

```cypher
MATCH (d:Disease {name: $disease_name})
MATCH (i:Ingredient)-[:CURES]->(d)
OPTIONAL MATCH (d)-[:MENTIONED_IN]->(h:Hadith)
OPTIONAL MATCH (r:Reference)-[:HAS_HADITH]->(h)
RETURN 
  i.name AS ingredient,
  h.name AS hadith_text,
  r.reference AS reference
LIMIT 20
```

**OUTPUT_FIELDS:**

- ingredient
- hadith_text
- reference

**NOTES:**
Core evidence query for your app

## B. Ingredient → Compound (WITH SOURCE)

**USE_CASE_NAME:**

Ingredient → Compounds WITH Source + Metadata

**DESCRIPTION:**
Get compounds + source + quantity

**INPUTS:**
ingredient_name

**CYPHER_QUERY:**

```cypher
MATCH (i:Ingredient {name: $ingredient_name})
MATCH (i)-[r:CONTAINS]->(c:ChemicalCompound)
RETURN 
  c.name AS compound,
  r.source AS source,
  r.quantity AS quantity,
  r.unit AS unit,
  r.food_part AS food_part
LIMIT 50
```

**OUTPUT_FIELDS:**

- compound
- source
- quantity
- unit
- food_part

## C. Ingredient → Compound → DrugCompound → Drug (FULL TRACE)

**USE_CASE_NAME:**

Ingredient → Drug Mapping WITH SOURCE TRACE

**DESCRIPTION:**
Full mapping chain including source of drug relation

**INPUTS:**
ingredient_name

**CYPHER_QUERY:**

```cypher
MATCH (i:Ingredient {name: $ingredient_name})
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
```

**OUTPUT_FIELDS:**

- drug
- compound
- mapping_strength
- ingredient_source
- drug_source

**NOTES:**
This is CRITICAL for explainability

## D. Drug → Book → Download Link

### Use Case 1

**USE_CASE_NAME:**

Drug Source → Book Download Link

**DESCRIPTION:**
Resolve source → book → link

**INPUTS:**
drug_name

**CYPHER_QUERY:**

```cypher
MATCH (d:Drug {name: $drug_name})-[:IS_IN_BOOK]->(b:Book)
RETURN 
  b.name AS book_name,
  b.link AS download_link
LIMIT 5
```

**OUTPUT_FIELDS:**

- book_name
- download_link

### Use Case 2

**USE_CASE_NAME:**

Drug via Source → Book (Indirect via relationship.source)

**DESCRIPTION:**
Match book name stored in relationship.source

**INPUTS:**
ingredient_name

**CYPHER_QUERY:**

```cypher
MATCH (i:Ingredient {name: $ingredient_name})
MATCH (i)-[:CONTAINS]->(c:ChemicalCompound)
MATCH (c)-[:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc:DrugChemicalCompound)
MATCH (d:Drug)-[r:CONTAINS]->(dcc)
MATCH (b:Book)
WHERE toLower(b.name) = toLower(r.source)
RETURN DISTINCT b.name AS book_name, b.link AS link
LIMIT 10
```

**OUTPUT_FIELDS:**

- book_name
- link

**NOTES:**
Handles your "source string → book node" mapping

## E. Disease → FULL EXPLAINABLE CHAIN (BEST QUERY)

**USE_CASE_NAME:**

Disease → Ingredient → Compound → Drug → Source → Book

**DESCRIPTION:**
Complete explainable pipeline

**INPUTS:**
disease_name

**CYPHER_QUERY:**

```cypher
MATCH (d:Disease {name: $disease_name})
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
```

**OUTPUT_FIELDS:**

- ingredient
- compound
- drug
- strength
- ingredient_source
- drug_source
- book_link

## F. Hadith Deep Linking

**USE_CASE_NAME:**

Hadith → Reference → Book

**DESCRIPTION:**
Full hadith source chain

**INPUTS:**
disease_name

**CYPHER_QUERY:**

```cypher
MATCH (d:Disease {name: $disease_name})-[:MENTIONED_IN]->(h:Hadith)
MATCH (r:Reference)-[:HAS_HADITH]->(h)
OPTIONAL MATCH (b:Book)
WHERE toLower(b.name) = toLower(r.reference)
RETURN 
  h.name AS hadith,
  r.reference AS reference,
  b.link AS book_link
LIMIT 10
```

## G. Source Validation Queries

**USE_CASE_NAME:**

Check if Source Exists as Book

```cypher
MATCH (b:Book)
WHERE toLower(b.name) = toLower($source_name)
RETURN count(b) > 0 AS exists
```

## H. Advanced Filtering

**USE_CASE_NAME:**

Exclude Weak + Keep Provenance

```cypher
MATCH (c:ChemicalCompound)-[r:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc)
RETURN c.name, dcc.name, type(r)
LIMIT 50
```

## I. Lightweight + Smart Queries

**USE_CASE_NAME:**

Count Drugs per Ingredient with only same chemical compounds

```cypher
MATCH (i:Ingredient {name: $ingredient_name})
MATCH (i)-[:CONTAINS]->(c)
MATCH (c)-[:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc)
MATCH (d:Drug)-[:CONTAINS]->(dcc)
RETURN count(DISTINCT d) AS drug_count
```

Count Drugs per Ingredient with  same chemical compounds + chemical compounds which may have similar name, so model have to decide using it's own knowledge, weather to consider these or not

```cypher
MATCH (i:Ingredient {name: $ingredient_name})
MATCH (i)-[:CONTAINS]->(c)
MATCH (c)-[:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO]->(dcc)
MATCH (d:Drug)-[:CONTAINS]->(dcc)
RETURN count(DISTINCT d) AS drug_count
```

Count Drugs per Ingredient with  same chemical compounds + chemical compounds which may have similar name as well as those chemical compounds which may or may not have similar name but a part of name could be similar so for considering these model have to be very carefull

```cypher
MATCH (i:Ingredient {name: $ingredient_name})
MATCH (i)-[:CONTAINS]->(c)
MATCH (c)-[:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc)
MATCH (d:Drug)-[:CONTAINS]->(dcc)
RETURN count(DISTINCT d) AS drug_count
```