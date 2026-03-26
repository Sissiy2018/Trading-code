# Some Forecasts That Might Be Useful

**Note:**  
The following data cannot always be extracted cleanly at once. You may see multiple rows for the same `Date + RIC`. This happens because each field can have different effective timestamps internally. You may also see many empty rows. Please specify clearly which fields you want.  

SmartEstimate is particularly useful. See the last section for more details.

---

## Price Target – Mean  
**Field:** `TR.PriceTargetMean`

**Definition:**  
The statistical average of all broker estimates determined to be on the majority accounting basis. Price Target is the projected price level forecasted by the analyst within a specific time horizon.

**Update Frequency:**  
Updated irregularly, whenever changes or updates occur. For some stocks this data is nearly daily. Median estimate is also available.

---

## EBITDA – Mean  
**Field:** `TR.EBITDAMean`

**Definition:**  
The statistical average of all broker estimates determined to be on the majority accounting basis. EBITDA gauges the raw earnings power of a company before debt servicing, corporate taxes, and any allowances made for depreciation and amortization costs. It is calculated in general form by taking the pre-tax corporate income of a company, adding back any depreciation and amortization costs charged, plus any interest expense on debt (subtracting any capitalized interest).

**Update Frequency:**  
Updated irregularly.

---

## EBIT – Mean  
**Field:** `TR.EBITMean`

**Definition:**  
The statistical average of all broker estimates determined to be on the majority accounting basis. EBIT represents the earnings of a company before interest expense and income taxes paid. It gauges corporate earnings before debt servicing to creditors (including bondholders) and the payment of corporate taxes. It is calculated in general form by taking the pre-tax corporate income of a company, adding back interest expense on debt, and subtracting any interest capitalized.

Similarly, mean forecasts are available for:
- Operating Income  
- Net Income  
- Capex  
- Cash Flow Per Share  
- ROE  
- ROA  

---

## Dividend Per Share – Mean  
**Field:** `TR.DPSMean`

**Definition:**  
The statistical average of all broker estimates determined to be on the majority accounting basis. Dividends Per Share are a corporation's common stock dividends on an annualized basis, divided by the weighted average number of common shares outstanding for the year. In the US, dividend per share is calculated before withholding taxes (though for some non-US companies DPS is calculated after withholding taxes).

**Update Frequency:**  
Estimated less often.

---

## Return On Equity – Mean  
**Field:** `TR.ROEMean`

**Definition:**  
The statistical average of all broker estimates determined to be on the majority accounting basis. Return On Equity is a profitability ratio calculated by dividing a company's net income by total equity of common shares.

Similarly, ROA (Return on Assets) estimates are available.

**Update Frequency:**  
Estimated less often.

---

## Earnings Per Share – Mean  
**Field:** `TR.EPSMean`

**Definition:**  
The statistical average of all broker estimates determined to be on the majority accounting basis. Earnings Per Share is defined as the EPS that the contributing analyst considers appropriate for valuing a security. This figure may include or exclude certain items depending on the contributing analyst's specific model.

---

## Recommendation – Mean (1–5)  
**Field:** `TR.RecMean`

**Definition:**  
Recommendation Numeric Mean based on the Standard Scale of:
- Strong Buy (1)  
- Buy (2)  
- Hold (3)  
- Sell (4)  
- Strong Sell (5)

---

## Recommendation – Mean Label  
**Field:** `TR.RecLabel`

**Definition:**  
Recommendation Mean Label based on the Standard Scale of:
- Strong Buy  
- Buy  
- Hold  
- Sell  
- Strong Sell  

Median recommendation is also available. These are estimated less often.

---

## Dividend Per Share – SmartEstimate®  
**Field:** `TR.DPSSmartEst`

**Definition:**  
The SmartEstimate® is an indicator of future earnings that improves upon the accuracy of the mean estimate by placing higher weight on recent forecasts and on top-rated analysts. Dividends Per Share are a corporation's common stock dividends on an annualized basis, divided by the weighted average number of common shares outstanding for the year. In the US, dividend per share is calculated before withholding taxes (though for some non-US companies DPS is calculated after withholding taxes).

---

## SmartEstimate Coverage

SmartEstimate is available for:
- Dividend Per Share (DPS)  
- Earnings Per Share (EPS)  
- Revenue / Sales  
- EBIT / EBITDA  
- Net Income  

If you need additional SmartEstimate data, search for **DIB** in Workspace.

---

## StarMine Analyst Revisions Model
**Field:** `TR.ARMIntraGlobalScore`, `TR.ARMIntraSectorScore`

A percentile rank based on the Analyst Revisions Model algorithm in which a security is compared to all other securities in the **same region and sector**/ all global secturities. Scores of 100 indicate the highest rated stocks in the sector, scores of 1 indicate the lowest rated stocks in the sector. ARM is highly predictive of relative price movement and is effective across stocks in each capitalization category, investment style, and market sector. Although ARM is a proprietary model, you can observe the key drivers on a stock by looking at individual revision components. ARM Intraday model **updates scores every hour-24 times a day**. Thus, whenever a security experiences an update to its SmartEstimate, the model recalculates its score.


---

## Earnings Quality Region Rank
**Field:** `TR.EQCountryListRank`

A percentile rank based on the Earnings Quality Model algorithm in which a security is compared to all other securities trading in the same region. Scores of 100 indicate the highest rated stocks in the region; scores of 1 indicate the lowest rated stocks in the region. **Earnings Quality: The degree to which past earnings are reliable and likely to persist.** StarMine's Earnings Quality model is a directional model that measures earnings sustainability across four factors: accruals, cash flow, operational efficiency and the company's use of exclusions.


---

I did not find **MarketPsych Analytics**, but I have found something useful, tell me if you want those data:

## Combined Alpha Model Region Rank 
**Field:** `TR.CombinedAlphaRegionRank`

The regional 1-100 percentile rank based on StarMine Combined Alpha Model, which combines the available StarMine models in the optimal, static, linear weighting scheme for each region. Scores of 100 indicate the highest rated stocks in the region; scores of 1 indicate the lowest rated stocks in the region. StarMine models included in the Combined Alpha Model are: Analyst Revisions, Earnings Quality, Relative Valuation, Intrinsic Valuation, Price Momentum, Smart Holdings, Insider Filings (US only), and Short Interest (US only).


## Credit Text Mining News Component
**Field:** `TR.CreditTextNewsComp`

The current global 1-100 rank of a company's credit riskiness based on textual data in Reuters news articles according to the StarMine Text Mining Credit Risk Model. Lower risk companies are given higher scores.

