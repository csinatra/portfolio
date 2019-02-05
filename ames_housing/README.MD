# Ames Housing Project

## Problem Statement

The goal of this project is to identify and optimize a production model based on the Ames, IA dataset that can be used as a proof of concept model for a consumer-grade appraisal tool. The input will be numeric and non-numeric home feature data that will be used to return an accurate sale price prediction for the given home. Model performance will be judged by RMSE and \$r^2 scoring on a test dataset after fitting an algorithm with training data. After refining this model, suggestions for generalization, implementation, as well as enterprise/commercial grade application will be reviewed.

## Executive Summary
I identified a model with ±\$22902.22 predictive performance, on par with other web-based consumer pricing tools [(Source)](https://realestate.usnews.com/real-estate/articles/7-online-tools-to-help-you-estimate-your-homes-value). This was based on 65 home features that can continue to be refined for implementation in an app environment. Since many of the features relate to fixed home characteristics, the actual UX flow will be largely automated due to availability of this data in municipal databases.

## Background

The Ames dataset is well known within the data-science community. By leveraging the predictive power of recursive model optimization processes we can identify trends across home features and geographic markets that are not immediately obvious even to seasoned professionals. Identifying the relative contribution these features make to overall home value will allow consumers and real estate professions work with greater confidence when listing properties as sellers and more quickly identify underpriced homes as buyers, resulting in higher profit margins or greater savings, resp. Additionally, by identifying home features that contribute more significantly to home value, both investors and home owners can better allocate resources for home restorations and other capital investments. This model can also serve as an additional validation tool for home appraisers or be offered as a consumer product in lieu of employing a professional appraiser.

According to U.S. News & World Report the top 7 consumer-grade home pricing tools can show over ±\$100,000 in variation in home valuation [(Source)](https://realestate.usnews.com/real-estate/articles/7-online-tools-to-help-you-estimate-your-homes-value). Given the wide availability of property data maintained by county clerks offices and other municipal offices, the potential predictive power of regression-based CMA is stronger than ever.

This project will outline key findings as well as provide business insights to guide further investigation.

## Data Dictionary
Refer to http://jse.amstat.org/v19n3/decock/DataDocumentation.txt

## Findings
In this project I found that the LassoCV model delivered the lowest RMSE (22902.21) and r2 score (90.57), narrowing predictions to \$22902.22. The next best model algorithm was RidgeCV, followed by ElasticNetCV, Linear Regression, and finally KNNeighbors. The Features of most interest were:

Beta 0:
- 180368.80

Positive features

| feature | weight |
| --- | --- |
| Gr Liv Area | 2.17e+04 |
| Overall Qual | 1.27e+04 |
| BsmtFin SF | 7.61e+03 |
| Kitchen Qual_Ex | 7.28e+03 |
| Year Built | 6.87e+03 |
| Total Bsmt SF | 6.38e+03 |
| Garage Area | 5.76e+03 |
| Bldg Type_1Fam | 5.30e+03 |
| Bsmt Qual_Ex | 5.21e+03 |
| Exter Qual_Ex | 4.80e+03 |
| Neighborhood_NridgHt | 4.58e+03 |
| Lot Area | 4.54e+03 |
| Neighborhood_StoneBr | 4.30e+03 |
| Overall Cond | 4.25e+03 |
| Bsmt Exposure_Gd | 4.07e+03 |
| Sale Type_New | 3.85e+03 |
| Neighborhood_NoRidge | 3.65e+03 |
| Mas Vnr Area | 3.15e+03 |
| Functional_Typ | 2.52e+03 |
| Foundation_PConc | 2.39e+03 |
| Fireplaces | 2.34e+03 |
| Exterior_Brk | 2.08e+03 |
| Year Remod/Add | 2.04e+03 |
| Neighborhood_Crawfor | 1.84e+03 |
| Roof Matl_WdShngl | 1.56e+03 |
| Neighborhood_Somerst | 1.31e+03 |
| Neighborhood_GrnHill | 1.29e+03 |
| Baths | 1.25e+03 |
| 1st Flr SF | 1.21e+03 |
| Exterior_Cmnt | 9.88e+02 |
| Bsmt Fin_GLQ | 9.78e+02 |
| Wood Deck SF | 7.05e+02 |
| Garage Qual_Gd | 5.71e+02 |
| Garage Qual_Ex | 4.89e+02 |
| Garage Cars | 4.49e+02 |
| Fireplace Qu_Gd | 4.24e+02 |
| Bsmt Baths | 3.45e+02 |
| Mas Vnr Type_Stone | 3.15e+02 |
| Neighborhood_BrkSide | 2.57e+02 |
| Land Contour_HLS | 2.55e+02 |
| Garage Type_BuiltIn | 2.39e+02 |
| Sale Type_Con | 1.95e+02 |
| Heating QC_Ex | 1.68e+01 |

Negative features

| feature | weight |
| --- | --- |
| Mas Vnr Type_BrkCmn | -7.31e+01 |
| Neighborhood_Edwards | -1.30e+02 |
| Garage Type_Basment | -1.84e+02 |
| Condition_RR | -2.76e+02 |
| Sale Type_ConLI | -2.82e+02 |
| Heating_OthW | -2.83e+02 |
| Bsmt Qual_TA | -3.11e+02 |
| Kitchen Qual_TA | -3.12e+02 |
| Bsmt Fin_LwQ | -3.27e+02 |
| Exterior_Plywood | -4.79e+02 |
| Functional_Maj1 | -5.74e+02 |
| Roof Style_Mansard | -7.78e+02 |
| Bsmt Exposure_No | -1.07e+03 |
| Sale Type_COD | -1.13e+03 |
| MS Zoning_RM | -1.28e+03 |
| Heating QC_TA | -1.51e+03 |
| Condition_Street | -1.59e+03 |
| Exter Qual_TA | -2.41e+03 |
| Misc Feature_Elev | -1.19e+04 |
| Roof Matl_ClyTile | -1.43e+04 |

## Business Recommendations
Since this model performs on par with existing consumer tools, I consider this a positive proof of concept. Next steps are continued model refinement as well as generalization to other home markets. Features that can be preloaded from municipal data can also be identified to develop a more streamlined UX environment and reduce the number of features that have to be manually assessed by the user.


## Conclusions
From my model comparisons I concluded that regularization methods such as Lasso and Ridge perform better than other modeling algorithms. This is likely due to many of the features having some level of colinearity that in other models would be outright excluded. For Lasso and Ridge they are still able to affect our predictions, however their impact is attenuated to avoid over-weighting. Since Lasso and Ridge both performed well, ElasticNet can probably be tuned to capture the best qualities of each model and increase predictive accuracy. 
