# Getting the ntuples

The data comes from the LHCb Open Data Ntupling Service (https://opendata-lhcb-ntupling-service.app.cern.ch, guide at https://lhcb-opendata-guide.web.cern.ch). Sign in, create a request, and in the decay search filter by stripping line: under `StrippingDstarCPForPromptCharm` tick all four `D*(2010)+ → (D0 → h+ h-) π+` entries, the D̄0-form ones included, because that line labels every D0 as +421 and the D̄0 entries are what capture the D*- candidates through their `]CC` clause. An earlier request with a single `]CC` row per mode delivered only the D*+ half of the sample, the D*- trees came back empty, which is why both forms are needed. Under `StrippingD2hhPromptDst2D2RSLine` tick `D*(2010)+ → (D0 → K- π+) π+` for the Kπ control. Name the trees `DstpD02KK`, `DstmD02KK`, `DstpD02PiPi`, `DstmD02PiPi`, `DstD02KPi`.

| setting | value |
|---|---|
| dataset | Collision16, CHARM.MDST, Stripping 28r2, MagDown and MagUp |
| tuple tools | Kinematic, Pid, ANNPID, Geometry, EventInfo, TrackInfo, Primaries, Propertime, TISTOS |
| request configuration as submitted | `docs/provenance/*.yaml` |

Ask for a test production first. When the test file arrives run `notebooks/05_validate_production_file.py` on it and do not confirm the full production unless every check passes, in particular the branching ratio check against the Kπ control, which is the one that catches the defect described in the README. Put the delivered files in `data/v3/`, list each one with its polarity in `config.PRODUCTIONS`, and paste the output of `tools/pin_checksums.py` into the same table. The production behind this repository is request 102 (productions 00412869 MagDown and 00412870 MagUp), four files, 151.9 pb⁻¹.
