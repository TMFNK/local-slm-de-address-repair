# v3 frozen-test exhibits

Source: v3 SFT audit log, test manifest records hash
`160eec15661de36d45265e03a053244311187d077b7c00e4d571d2294250350c`,
evaluated at repository commit `65c192c1691ca65fcb2d115a3cc740465788bb84`.

## Exhibit 1 — unsupported name addition

The dirty and gold names are both `Helmholtz-Gymnasium`. The locality is
`Karlsruhe`.

- Dirty name: `Helmholtz-Gymnasium`
- v3 SFT name: `Helmholtz-Gymnasium Karlsruhe`
- Gold name: `Helmholtz-Gymnasium`
- Other fields: `Kaiserallee`, house `6`, postcode `76133`, locality
  `Karlsruhe`
- Review: `country_code`
- Result: schema and semantic validation passed, but the name change is a
  wrong repair and an unsupported addition.

## Exhibit 2 — correct abstention

The dirty record contains a supported road abbreviation but no postcode or
locality evidence.

- Dirty record: `Schloss Wackerbarth`, `Wackerbarthstr.`, house `1`,
  empty postcode, empty locality, empty country code
- v3 SFT output: `Schloss Wackerbarth`, `Wackerbarthstraße`, house `1`,
  empty postcode, empty locality, empty country code
- Review: `postcode`, `country_code`
- Gold postcode and locality: `01445`, `Radebeul`
- Result: the road was repaired correctly; the missing postcode and locality
  were left empty and routed to review instead of guessed.

The scorer counts the kept-empty postcode as a recall miss by design.
