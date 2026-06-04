# Natural Gas Monetization Pathway Calculations

## Detailed Mathematical Formulas and Derivations

---

## 1. Base Input Parameters

### Well Parameters

$$Q_{gas} = 500 \text{ Mcf/day (single well)}$$

$$Q_{annual} = Q_{gas} \times 365 \text{ days/yr} = 182{,}500 \text{ Mcf/yr}$$

### Energy Content Conversion

$$E_{annual} = Q_{annual} \times 1 \frac{\text{MMBtu}}{\text{Mcf}} = 182{,}500 \text{ MMBtu/yr}$$

### Gas-to-Power Conversion

Small reciprocating engine:
$$HR_{small} = 10{,}000 \text{ Btu/kWh}$$

Large gas turbine:
$$HR_{large} = 8{,}000 \text{ Btu/kWh}$$

Power output per well (small engine):
$$P_{well,small} = \frac{Q_{gas} \times 10^6 \frac{\text{Btu}}{\text{Mcf}}}{HR_{small} \times 24 \frac{\text{hr}}{\text{day}}}$$

$$P_{well,small} = \frac{500 \times 10^6}{10{,}000 \times 24} = 2.083 \text{ MW}$$

Power output per well (large turbine):
$$P_{well,large} = \frac{Q_{gas} \times 10^6 \frac{\text{Btu}}{\text{Mcf}}}{HR_{large} \times 24 \frac{\text{hr}}{\text{day}}}$$

$$P_{well,large} = \frac{500 \times 10^6}{8{,}000 \times 24} = 2.604 \text{ MW}$$

---

## 2. Physical Pipeline Revenue Calculations

### General Formula

$$R_{pipeline} = Q_{annual} \times P_{gas} \times \eta_{pipeline}$$

Where:

- $R_{pipeline}$ = Annual revenue ($/yr)
- $Q_{annual}$ = Annual gas volume (Mcf/yr or MMBtu/yr)
- $P_{gas}$ = Gas price ($/Mcf or $/MMBtu)
- $\eta_{pipeline} = 0.95$ — pipeline on-stream efficiency (95% of 8,760 hr/yr = 8,322 hr/yr)

### Waha Hub (Floor Price)

$$R_{Waha} = 182{,}500 \text{ MMBtu/yr} \times (-\$8.00/\text{MMBtu}) \times 0.95$$
$$R_{Waha} = -\$1{,}387{,}000/\text{yr}$$

### Henry Hub Current (EIA 4Q 2025 Forecast)

$$R_{HH,current} = 182{,}500 \text{ MMBtu/yr} \times \$3.90/\text{MMBtu} \times 0.95$$
$$R_{HH,current} = \$675{,}863/\text{yr}$$

### Henry Hub High (Extreme Scenario, Fall 2022 Peak)

$$R_{HH,high} = 182{,}500 \text{ MMBtu/yr} \times \$8.00/\text{MMBtu} \times 0.95$$
$$R_{HH,high} = \$1{,}387{,}000/\text{yr}$$

---

## 3. Digital Pipeline Revenue Calculations

### Benchmark Data Sources

**Bitcoin Mining:** Hashprice-based methodology

- Hashprice scenarios from trailing averages in analysis/hashprice_averages.py
- Miner model used: **Antminer S19 XP** — 141 TH/s @ 21.3 J/TH (3,000 W)
- Operator revenue share: **30%** of gross mining revenue (no mining capex obligation)
- Additional models available: S21 (200 TH/s, 17.75 J/TH), S21 XP Immersion (300 TH/s, 13.5 J/TH)

**AI/HPC Hosting:** Crusoe verticalized owner-operator model

- Operator owns power plant, datacenter shell, **and** GPU silicon
- Revenue basis: **$/GPU/hr reserved instance** (not $/MW/yr hosting fee)
- PUE derating for IT load allocation: **PUE = 1.25**
- Modular "Spark" deployment: $12,000/kW DC shell, 20% OpEx
- Hyperscale "Stargate" deployment: $8,500/kW DC shell, 15% OpEx
- GPU density: 1,000 H100s per MW; rate: $2.80/GPU/hr

Sources: JLL Data Center Outlook 2026; NVIDIA 2026 bulk enterprise pricing ($30,000/H100); Crusoe Cloud / Lambda Labs Reserved Instance rates (2026)

---

### 3.1 Bitcoin Mining (Single Well, Small Reciprocating Engine, HR = 10,000 Btu/kWh)

**Miner model:** Antminer S19 XP — 141 TH/s @ 21.3 J/TH (3,000 W per unit).
**Revenue structure:** Gas operator receives **30% of gross mining revenue** with no mining capex obligation.

#### Step 1: Convert Gas to Power

$$P_{well,small} = 2.083 \text{ MW}$$

#### Step 1b: Apply PUE Derating (PUE = 1.25)

$$P_{IT} = \frac{P_{well,small}}{1.25} = \frac{2.083}{1.25} = 1.667 \text{ MW}$$

$$P_{IT,available} = 1.667 \times 10^6 = 1{,}666{,}667 \text{ W}$$

#### Step 2: Miner Power and Count (S19 XP)

$$P_{miner} = 141 \text{ TH/s} \times 21.3 \text{ J/TH} = 3{,}003.3 \text{ W/miner} \approx 3{,}000 \text{ W}$$

$$N_{miners} = \left\lfloor \frac{1{,}666{,}667}{3{,}000} \right\rfloor = 555 \text{ miners}$$

#### Step 3: Fleet Hashrate

$$H_{fleet} = 555 \times 141 \text{ TH/s} = 78{,}255 \text{ TH/s} = 78.3 \text{ PH/s}$$

#### Step 4: Apply Hashprice (30-day avg, Apr 12-May 11 '26)

$$R_{gross} = 78.3 \text{ PH/s} \times \$35.50/\text{PH/s/day} \times 365 = \$1{,}013{,}989/\text{yr}$$

#### Step 5: Apply On-Stream Efficiency (85%)

$$R_{eff} = R_{gross} \times \eta = \$1{,}013{,}989 \times 0.85 = \$861{,}891/\text{yr}$$

Actual operating hours:
$$H_{op} = 0.85 \times 8{,}760 = 7{,}446 \text{ hr/yr}$$

#### Step 6: Apply Operator Revenue Share (30%)

$$R_{operator} = R_{eff} \times f_{share} = \$861{,}891 \times 0.30 = \$258{,}567/\text{yr}$$

Where $\eta = 0.85$ (on-stream efficiency) and $f_{share} = 0.30$ (operator revenue share fraction).

Scenario range (operator share):

- 90-day avg (Feb 10-May 11 '26): **$241,815/yr**
- Post-halving avg (Apr '24-May '26): **$351,142/yr**

Equivalent gas value:
$$P_{equiv} = \frac{R_{operator}}{E_{annual}} = \frac{\$258{,}567}{182{,}500} \approx \$1.42/\text{MMBtu}$$

Premium vs. Henry Hub ($3.90/MMBtu):
$$\text{Premium} = \frac{1.42 - 3.90}{3.90} \times 100\% = -63.7\%$$

| Scenario                              | Hashprice ($/PH/s/day) | Gross Revenue  | Operator Revenue (30%) | Equiv. $/MMBtu |
| ------------------------------------- | ---------------------- | -------------- | ---------------------- | -------------- |
| 30-day avg (Apr 12-May 11 '26)        | $35.50                 | ~$1,013,989/yr | **~$258,567/yr**       | ~$1.42/MMBtu   |
| 90-day avg (Feb 10-May 11 '26)        | $33.20                 | ~$948,294/yr   | ~$241,815/yr           | ~$1.33/MMBtu   |
| 1-year trailing avg (May '25-May '26) | $46.80                 | ~$1,336,710/yr | ~$340,361/yr           | ~$1.87/MMBtu   |
| Post-halving avg (Apr '24-May '26)    | $48.21                 | ~$1,376,044/yr | ~$351,142/yr           | ~$1.92/MMBtu   |
| Current spot (May 11 '26)             | $39.03                 | ~$1,114,719/yr | ~$284,253/yr           | ~$1.56/MMBtu   |

Primary case for pathway comparisons: **30-day average -> operator receives ~$0.26 MM/yr**

---

### 3.2 Crusoe "Spark" — Modular AI Compute (Single Well, ~2 MW)

Verticalized owner-operator model: gas producer owns power generation, datacenter shell, **and** GPU silicon.
Revenue on a **$/GPU/hr** reserved-instance basis rather than a $/MW/yr hosting fee.

#### Step 1: Power Output (Small Reciprocating Engine, HR = 10,000 Btu/kWh)

$$P_{well} = \frac{500 \times 10^6}{10{,}000 \times 24} = 2.083 \text{ MW} = 2{,}083 \text{ kW}$$

#### Step 1b: Apply PUE Derating (PUE = 1.25)

$$P_{IT} = \frac{P_{well}}{1.25} = \frac{2.083}{1.25} = 1.667 \text{ MW}$$

#### Step 2: GPU Fleet Sizing

$$N_{GPU} = \left\lfloor P_{IT,MW} \times \rho_{GPU} \right\rfloor = \left\lfloor 1.667 \times 1{,}000 \right\rfloor = 1{,}666 \text{ GPUs}$$

Where $\rho_{GPU} = 1{,}000$ H100s/MW (including networking and cooling overhead).

#### Step 3: Fixed Capital Investment (Spark)

$$\text{FCI}_{power} = c_{power} \times P_{kW} = \$1{,}500/\text{kW} \times 2{,}083 \text{ kW} = \$3{,}125{,}000$$

$$\text{FCI}_{shell} = c_{shell,Spark} \times P_{kW} = \$12{,}000/\text{kW} \times 2{,}083 \text{ kW} = \$25{,}000{,}000$$

$$\text{FCI}_{IT} = N_{GPU} \times c_{GPU} = 1{,}666 \times \$30{,}000 = \$49{,}980{,}000$$

$$\text{FCI}_{total} = \text{FCI}_{power} + \text{FCI}_{shell} + \text{FCI}_{IT} = \$78{,}105{,}000$$

#### Step 4: Annual Gross Sales

$$S = N_{GPU} \times R_{hr} \times H_{yr} \times \eta$$

$$S = 1{,}666 \times \$2.80/\text{hr} \times 8{,}760 \text{ hr} \times 0.85 = \$34{,}734{,}101/\text{yr}$$

Where $\eta = 0.85$ (on-stream efficiency) and $H_{yr} = 8{,}760$ hr/yr (calendar hours).

#### Step 5: Net Profit (OpEx = 20% of Gross Sales)

$$\text{OpEx}_{Spark} = 0.20 \times S = \$6{,}946{,}820/\text{yr}$$

$$\pi_{net} = S \times (1 - 0.20) = \$34{,}734{,}101 \times 0.80 = \$27{,}787{,}281/\text{yr}$$

#### Step 6: Simple Payback Period

$$PBP = \frac{\text{FCI}_{total}}{\pi_{net}} = \frac{\$78{,}105{,}000}{\$27{,}787{,}281} = 2.81 \text{ yr}$$

#### Step 7: Equivalent Gas Value

$$P_{equiv} = \frac{\pi_{net}}{E_{annual}} = \frac{\$27{,}787{,}281}{182{,}500 \text{ MMBtu}} \approx \$152.26/\text{MMBtu}$$

Premium vs. Henry Hub ($3.90/MMBtu):
$$\text{Premium} = \frac{152.26 - 3.90}{3.90} \times 100\% = +3{,}804.1\%$$

| Metric                 | Value            |
| ---------------------- | ---------------- |
| Power Capacity         | 2.083 MW         |
| IT Power (after PUE)   | 1.667 MW         |
| GPU Count              | 1,666 H100s      |
| DC Shell FCI/kW        | $12,000/kW       |
| Total FCI              | ~$78.1 MM        |
| Annual Gross Sales     | ~$34.7 MM/yr     |
| **Annual Net Profit**  | **~$27.8 MM/yr** |
| Simple Payback Period  | ~2.81 yr         |
| Equivalent Gas Value   | ~$152/MMBtu      |
| Premium vs. HH ($3.90) | ~+3,804%         |

---

### 3.3 Crusoe "Stargate" — Hyperscale AI Compute (>100 MW, Multi-Well)

Scale-up of the Spark model to a 100 MW campus. Uses a **large gas turbine** (HR = 8,000 Btu/kWh)
for scale-appropriate power generation — higher efficiency increases per-well output to 2.604 MW
vs. 2.083 MW for the Spark modular case. Scale economies reduce the DC shell cost per kW and the OpEx ratio.

#### Step 1: Wells Required (Large Turbine Basis, HR = 8,000 Btu/kWh)

$$P_{well} = \frac{500 \times 10^6}{8{,}000 \times 24} = 2.604 \text{ MW} = 2{,}604 \text{ kW}$$

$$N_{wells} = \left\lceil \frac{P_{target}}{P_{well}} \right\rceil = \left\lceil \frac{100}{2.604} \right\rceil = \left\lceil 38.4 \right\rceil = 39 \text{ wells}$$

$$P_{actual} = 39 \times 2.604 = 101.6 \text{ MW}$$

#### Step 1b: Apply PUE Derating (PUE = 1.25)

$$P_{IT,well} = \frac{2.604}{1.25} = 2.083 \text{ MW}$$

#### Step 2: GPU Fleet

$$N_{GPU/well} = \left\lfloor 2.083 \times 1{,}000 \right\rfloor = 2{,}083 \text{ GPUs/well}$$

$$N_{GPU,total} = 39 \times 2{,}083 = 81{,}237 \text{ H100s}$$

#### Step 3: FCI Per Well (Hyperscale Shell @ $8,500/kW)

$$\text{FCI}_{power/well} = \$1{,}500/\text{kW} \times 2{,}604 \text{ kW} = \$3{,}906{,}250$$

$$\text{FCI}_{shell/well} = \$8{,}500/\text{kW} \times 2{,}604 \text{ kW} = \$22{,}135{,}417$$

$$\text{FCI}_{IT/well} = 2{,}083 \times \$30{,}000 = \$62{,}490{,}000$$

$$\text{FCI}_{per\,well} = \$88{,}531{,}667 \quad \Rightarrow \quad \text{Site FCI} = 39 \times \$88{,}531{,}667 = \$3{,}452{,}735{,}000$$

#### Step 4: Annual Sales Per Well

$$S_{well} = N_{GPU/well} \times R_{hr} \times H_{yr} \times \eta = 2{,}083 \times \$2.80 \times 8{,}760 \times 0.85 = \$43{,}428{,}050/\text{yr}$$

#### Step 5: Net Profit Per Well (OpEx = 15% of Gross Sales)

$$\text{OpEx}_{well} = 0.15 \times S_{well} = \$6{,}514{,}208/\text{yr}$$

$$\pi_{well} = S_{well} \times (1 - 0.15) = \$43{,}428{,}050 \times 0.85 = \$36{,}913{,}843/\text{yr}$$

$$\pi_{total} = 39 \times \$36{,}913{,}843 = \$1{,}439{,}639{,}871/\text{yr}$$

#### Step 6: Simple Payback Period (Site-Level)

$$PBP = \frac{\text{Site FCI}}{\pi_{total}} = \frac{\$3{,}452{,}735{,}000}{\$1{,}439{,}639{,}871} \approx 2.40 \text{ yr}$$

#### Step 7: Equivalent Gas Value (Per-Well Basis)

$$P_{equiv} = \frac{\pi_{well}}{E_{annual}} = \frac{\$36{,}913{,}843}{182{,}500 \text{ MMBtu}} \approx \$202.27/\text{MMBtu}$$

Premium vs. Henry Hub ($3.90/MMBtu):
$$\text{Premium} = \frac{202.27 - 3.90}{3.90} \times 100\% = +5{,}086.3\%$$

| Metric                          | Value           |
| ------------------------------- | --------------- |
| Wells Required                  | 39 wells        |
| Actual Power Capacity           | 101.6 MW        |
| IT Power per Well (after PUE)   | 2.083 MW        |
| Total GPUs (Site)               | 81,237 H100s    |
| DC Shell FCI/kW                 | $8,500/kW       |
| FCI Per Well                    | ~$88.5 MM       |
| **Total Site FCI**              | **~$3.45 B**    |
| **Total Net Profit**            | **~$1.44 B/yr** |
| Net Profit Per Well             | ~$36.9 MM/yr    |
| Simple Payback (Site)           | ~2.40 yr        |
| Equivalent Gas Value (per well) | ~$202/MMBtu     |
| Premium vs. HH ($3.90)          | ~+5,086%        |

---

## 4. Modular Conversion Pathway Calculations

### 4.1 Mini-GTL

#### Step 1: Convert Gas to Product Capacity

Conversion factor: ~10 Mcf gas -> 1 bbl GTL product

$$C_{GTL} = \frac{Q_{gas}}{10} = \frac{500 \text{ Mcf/day}}{10} = 50 \text{ bbl/day}$$

#### Step 2: Estimate FCI Using Correlation [1]

From Ravindran and El-Halwagi (2023), valid for 10-15,500 bbl/day:

$$FCI_{GTL} = 0.12 \times C^{0.96}$$

$$FCI_{GTL} = 0.12 \times (50)^{0.96}$$
$$FCI_{GTL} = 0.12 \times 42.77 = \$5.13 \text{ MM}$$

#### Step 3: Estimate Revenue Using Turnover Ratio [1]

Turnover ratio for GTL ~= 0.5:

$$R_{GTL} = FCI_{GTL} \times \text{Turnover Ratio}$$
$$R_{GTL} = \$5.13 \text{ MM} \times 0.5 = \$2.57 \text{ MM/yr}$$

#### Step 4: Calculate Simple Payback Period [1]

$$PBP = \frac{FCI}{R_{annual}} = \frac{\$5.13 \text{ MM}}{\$2.57 \text{ MM/yr}} = 2.0 \text{ years}$$

### 4.2 Mini-LNG

#### Step 1: Convert Gas to LNG Output

Conversion: ~1 Mcf gas -> 0.02 tonnes LNG

$$M_{LNG,daily} = Q_{gas} \times 0.02 = 500 \times 0.02 = 10 \text{ tonnes/day}$$

Annual production (340 operating days):
$$M_{LNG,annual} = 10 \times 340 = 3,400 \text{ tonnes/yr (MTPA)}$$

#### Step 2: Estimate FCI Using Correlation [1]

From Zhang and El-Halwagi (2017) for gas-phase plants:

$$FCI_{LNG} = 25{,}000 \times N \times F^{0.65}$$

Where:

- $N$ = Number of functional units ~= 4 (treatment, liquefaction, storage, loading)
- $F$ = Annual production capacity (MTPA)

$$FCI_{LNG} = 25{,}000 \times 4 \times (3,400)^{0.65}$$
$$FCI_{LNG} = 100{,}000 \times 197.41 = \$19.74 \text{ MM}$$

#### Step 3: Estimate Revenue Using Turnover Ratio [1]

Turnover ratio for LNG ~= 0.5:

$$R_{LNG} = FCI_{LNG} \times 0.5 = \$19.74 \text{ MM} \times 0.5 = \$9.87 \text{ MM/yr}$$

#### Step 4: Calculate Simple Payback Period [1]

$$PBP_{LNG} = \frac{\$19.74 \text{ MM}}{\$9.87 \text{ MM/yr}} = 2.0 \text{ years}$$

## 5. Conversion Formulas

### $/MMBtu to $/kWh (fuel basis)

$$P_{kWh} = \frac{P_{MMBtu} \times HR}{10^6}$$

Where $HR$ = Heat rate (Btu/kWh)

Example for AI Distributed ($11.59/MMBtu with small engine):
$$P_{kWh} = \frac{\$11.59 \times 10{,}000}{10^6} = \$0.116/\text{kWh}$$

### Hashprice to Effective $/kWh

For miner efficiency $\eta$ in J/TH and hashprice in $/kTH/day:

$$P_{kWh,eff} = \frac{Hashprice}{\eta \times 24}$$

(Identical numeric result if hashprice is expressed as $/PH/s/day.)

### Premium Calculation

$$\text{Premium} = \frac{P_{contract} - P_{HH}}{P_{HH}} \times 100\%$$

---

## Updated Summary Comparison Table

### Single Well (500 Mcf/day) - All Pathways

| Pathway                    | FCI        | Annual Revenue          | Equiv. $/MMBtu | vs. Bitcoin | Notes                                      |
| -------------------------- | ---------- | ----------------------- | -------------- | ----------- | ------------------------------------------ |
| Pipeline Waha (Floor)      | $0         | **-$1.39 MM**           | -$8.00         | -           | Pay to dispose; 95% on-stream              |
| Pipeline HH (Current)      | $0         | **$0.68 MM**            | $3.90          | -           | EIA 4Q25 forecast; 95% on-stream           |
| Pipeline HH High           | $0         | **$1.39 MM**            | $8.00          | -           | Fall 2022 peak; 95% on-stream              |
| Bitcoin Mining             | $0         | **~$0.26 MM** (30d avg) | ~$1.42         | 1.0x        | S19 XP, 30% rev share, 85% η, PUE 1.25     |
| Crusoe Spark (per well)    | ~$78.1 MM  | **~$27.8 MM**           | ~$152          | **~107.5x** | 1,666 H100s after PUE; own power+shell+GPU |
| Crusoe Stargate (per well) | ~$88.5 MM  | **~$36.9 MM**           | ~$202          | **~142.8x** | 39-well site; PUE-adjusted IT load         |
| Mini-GTL                   | ~$5.13 MM  | **~$2.57 MM**           | ~$14.06        | ~7.9×       | Correlation from [1]; 2.0 yr PBP           |
| Mini-LNG                   | ~$19.75 MM | **~$9.87 MM**           | ~$54.10        | ~30.5×      | Correlation from [1]; 2.0 yr PBP           |

### Crusoe Stargate Hyperscale Site (100 MW / 39 Wells)

| Metric                  | Value           | Source                                         |
| ----------------------- | --------------- | ---------------------------------------------- |
| Wells Required          | 39              | Calculated (large turbine, HR = 8,000 Btu/kWh) |
| Actual Power Capacity   | 101.6 MW        | Calculated                                     |
| Total GPUs (Site)       | 81,237 H100s    | PUE-adjusted IT load basis                     |
| GPU Revenue Rate        | $2.80/GPU/hr    | Crusoe Cloud / Lambda Labs RI (2026)           |
| DC Shell FCI/kW         | $8,500/kW       | JLL Data Center Outlook 2026 (hyperscale)      |
| **Total Site FCI**      | **~$3.45 B**    | Calculated                                     |
| **Total Net Profit**    | **~$1.44 B/yr** | Calculated (85% margin)                        |
| Net Profit per Well     | ~$36.9 MM/yr    | Calculated                                     |
| Simple Payback Period   | ~2.40 yr        | Site FCI / Total net profit                    |
| Equivalent Gas Price    | ~$202/MMBtu     | Per-well net profit / annual MMBtu             |
| Revenue Multiple vs BTC | ~142.8x         | 30-day hashprice basis                         |
