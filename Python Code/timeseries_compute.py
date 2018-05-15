import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

path = os.path.dirname(os.path.abspath(""))
path_latex = os.path.dirname(path) + "\Dissertation LaTeX Files"
sns.set(color_codes=True)
for i in range(2):
    if i == 1:
        file = "\Model Output\model_output_10yr.csv"
        graph = '\images\mbar_line_10yr.eps'
        rf = "10yr"
    else:
        file = "\Model Output\model_output_30yr.csv"
        graph = '\images\mbar_line_30yr.eps'
        rf = "30yr"

    # Reading the data into a dataframe from our csv files.
    df = pd.read_csv(path + file, sep=",",
                     dtype={"fyear": str, "ticker": str, "firmname": str, "industry": str, "sector": str,
                            "statevariable": float, "cash": float, "fixedcosts": float, "couponrate": float,
                            "liabilities": float, "equityobserved": float})

    df['fyear'] = pd.to_datetime(df['fyear'], format="%Y-%m-%d").dt.date
    df['year'] = pd.to_datetime(df['fyear']).dt.year

    # count total number of values per year
    df_count = pd.value_counts(df["year"])
    df_count = pd.Series.to_frame(df_count, name="count")
    df_count["year"] = df_count.index
    df_count = pd.DataFrame.sort_index(df_count)

    # compute indexed weighted average mbar
    years = df.year.unique()
    mbar_indexed_dict = {}
    total_eq_dict = {}

    # compute base weights
    baseweights = df.groupby("year", as_index=False)["equityobserved"].sum()
    baseweights = baseweights.rename(index=str, columns={"equityobserved": "total_equity"})
    df = pd.merge(baseweights, df, on=['year'])
    df["base_weight"] = df["equityobserved"] / df["total_equity"]

    # compute weighted mbar values
    df["base_weight_mbar"] = df["mbar"] * df["base_weight"]

    for year in years[1:]:
        # create dataframes with values in t and t+1
        df_t = df.loc[df["year"].isin([year-1])]
        df_tplus1 = df.loc[df["year"].isin([year])]

        # merge t and t+1 dataframes and reset index - setting up to compare
        df_compare = pd.concat([df_tplus1, df_t])
        df_compare = df_compare.reset_index(drop=True)

        # group by ticker and find all ticker values that occur twice then reindex based on the findings
        df_gpby = df_compare.groupby('ticker')
        idx = [x[0] for x in df_gpby.groups.values() if len(x) == 2]
        df_compare = df_compare.reindex(idx)

        # compute total equity per year
        df_sum = df_compare.groupby("year", as_index=False)["equityobserved"].sum()
        df_sum = df_sum.values.tolist()
        df_sum = df_sum[0][1]
        total_eq_dict.update({year: df_sum})

    # update the total_equity column to have the total equity of just comparable firms
    total_equity = pd.Series(total_eq_dict, name='total_equity')
    total_equity = total_equity.append(pd.Series(df.iloc[0]["total_equity"], index=[1998], name='total_equity'))
    total_equity = pd.Series.to_frame(total_equity)
    total_equity.index.name = "year"
    total_equity = total_equity.reset_index()
    total_equity = total_equity.sort_values('year', ascending=True)
    total_equity = total_equity.reset_index(drop=True)

    # merge and clean up original df
    df = pd.merge(total_equity, df, on=['year'])
    del df['total_equity_y']
    df = df.rename(index=str, columns={"total_equity_x": "total_equity"})

    # compute filtered weights per firm per year
    df["weight"] = df["equityobserved"] / df['total_equity']
    df["weighted_mbar"] = df["mbar"] * df["weight"]

    for t in years[1:]:
        if t - 1 == 1998:
            index_t0 = df.groupby("year", as_index=False)["base_weight_mbar"].sum()
            index_t0 = index_t0.values.tolist()
            index_t0 = index_t0[0][1]
            index_t = index_t0
        else:
            index_t = mbar_indexed_dict[t - 1]

        # create dataframes with values in t and t+1
        df_t = df.loc[df["year"].isin([t - 1])]
        df_tplus1 = df.loc[df["year"].isin([t])]

        # merge t and t+1 dataframes and reset index - setting up to compare
        df_compare = pd.concat([df_tplus1, df_t])
        df_compare = df_compare.reset_index(drop=True)

        # group by ticker and find all ticker values that occur twice then reindex based on the findings
        df_gpby = df_compare.groupby('ticker')
        idx = [x[0] for x in df_gpby.groups.values() if len(x) == 2]
        df_compare = df_compare.reindex(idx)

        # sum the weighted_mbar column to calculate average mbar for all firms in common between y0 and y1
        index_tplus1 = df_compare.groupby("year", as_index=False)["weighted_mbar"].sum()
        index_tplus1 = index_tplus1.values.tolist()
        index_tplus1 = index_tplus1[0][1]

        # calculate the % variation in the mbar values and compute indexed mbar
        variation = (index_tplus1 - index_t) / index_t
        indexed_mbar = index_t * (1 + variation)
        # print("t:", index_t, "t+1:", index_tplus1, "indexed mbar:", indexed_mbar, "variation:", variation)
        mbar_indexed_dict.update({t: indexed_mbar})

    # compute weighted average mbar (for showing the difference)
    avg_mbar = df.groupby("year", as_index=False)["weighted_mbar"].sum()
    avg_mbar = avg_mbar.values.tolist()
    avg_mbar2 = [item[1] for item in avg_mbar]

########################################################################################################################

    #  clear plot graphics to create new plot, set size and tick positions
    plt.gcf().clear()
    plt.figure(figsize=(8, 4))
    tickpos = range(1998, 2018, 2)

    # set x and y for sample size plot
    X = df_count["year"].tolist()
    y = df_count["count"].tolist()

    # create a histogram for datapoints per year
    plt.plot(X, y, linewidth=2, label="Firms in sample.", color='b')

    # plot an indicator of maximum and min value
    max_point = max(y)
    max_year = 2013

    plt.plot(max_year, max_point, "o", color='g', label="Maximum firm count")
    plt.text((max_year - 0.5), (max_point - 2.5), str(max_point), fontsize=10)

    min_point = min(y)
    min_year = 1998

    plt.plot(min_year, min_point, "o", color='r', label="Minimum firm count")
    plt.text((min_year + 0.5), min_point, str(min_point), fontsize=10)

    # set plot labels and ticks
    plt.xlabel("Years")
    plt.ylabel("Firm Count")
    plt.xticks(tickpos)

    plt.legend()
    plt.savefig(path_latex + '\images\\firm_count.eps', format='eps', dpi=1000)
    plt.savefig(path_latex + '\images\\firm_count.png', format='png', dpi=1000)


########################################################################################################################

    #  clear plot graphics to create new plot, set size and tick positions
    plt.gcf().clear()
    plt.figure(figsize=(9, 4))
    tickpos = range(1998, 2018, 2)

    # create m_bar line plot
    plt.xticks(tickpos)
    years = list(mbar_indexed_dict.keys())
    mbar = list(mbar_indexed_dict.values())
    mbar_2 = avg_mbar2[1:]

    max_point = max(mbar)
    max_year = 2008

    plt.plot(years, mbar, linewidth=2, color='b', label="Indexed m_bar values.")
    # plt.plot(years, mbar_2, linewidth=2, color='r', linestyle='dashed', label='non-indexed')

    plt.plot(max_year, max_point, "o", color='g', label="Maximum m_bar value.")
    plt.text((max_year + 0.5), (max_point - 0.005), str(round(max_point, 4)), fontsize=10)

    plt.legend()
    plt.savefig(path_latex + graph, format='eps', dpi=1000)
    plt.savefig(path_latex + '\images\\mbar_miu' + rf + ".png", format='png', dpi=1000)

########################################################################################################################
