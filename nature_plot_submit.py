import pandas as pd
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from lmfit import Model  # non-linear optimization by Levenberg-Marquardt algorithm
import nature_model_submit

data_path = "nature-data/pneumonia_panel_296_cities(submit).csv"
days = 27  # from Jan. 24 to Feb. 19


def exponential_logistic_for_plot(X, gamma, omega, alpha, beta_1):
    return (alpha * np.exp(beta_1 * X[0])) / (1 + np.exp(-1 * gamma * X[1] + omega))


def plot_dynamic_model_performance_3d():
    panel = pd.read_csv(data_path)
    panel = panel[panel["day"] <= 27]
    days = 27
    panel.fillna(0, inplace=True)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15, azim=-35)
    data = panel
    x = data["day"]
    y = data["wuhan_outflow"]
    z = data["confirmed"]
    y = np.log(y)
    X = np.vstack((y, x))
    model = Model(exponential_logistic_for_plot)
    params = model.make_params(gamma=0, omega=0, alpha=0, beta_1=2)
    result = model.fit(z, X=X, params=params)
    best_params = result.params  # best fitted parameters
    y_pred = result.best_fit
    data["confirmed_pred"] = y_pred
    print(r2_score(z, y_pred))
    data["error"] = data["confirmed_pred"] - data["confirmed"]
    df = data[data["error"] < 0]
    x = df["day"]
    y = df["wuhan_outflow"]
    z = df["confirmed"]
    y = np.log(y)
    ax.scatter(x, y, z, s=50, marker=".", c="#DC143C")
    df = data[data["error"] >= 0]
    x = df["day"]
    y = df["wuhan_outflow"]
    z = df["confirmed"]
    y = np.log(y)
    ax.scatter(x, y, z, s=50, marker=".", c="k")
    _y = np.linspace(0, 15, 100)
    _x = np.linspace(1, days + 1, days)
    x, y = np.meshgrid(_x, _y)
    z = (best_params["alpha"] * np.exp(best_params["beta_1"] * y)) / (
            1 + np.exp(-1 * best_params["gamma"] * x + best_params["omega"]))
    ax.plot_surface(x, y, z, cmap="rainbow", alpha=0.65)
    ax.set_ylabel("Outflow population from Wuhan(log)", fontsize=14)
    ax.set_xlabel("Time", fontsize=14, rotation=-25)
    ax.set_zlim(zmin=0, zmax=4000)
    plt.xticks(np.arange(1, days + 1, 2))
    ax.set_zlabel("Confirmed cases", fontsize=16, labelpad=10)
    plt.xlim(1, 27)
    plt.ylim(0, 15)
    plt.tight_layout()
    plt.savefig("nature-output/Figure-1(China).png")
    plt.close()
    data = panel[panel["province_en"] == "Hubei"]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15, azim=-35)
    x = data["day"]
    y = data["wuhan_outflow"]
    z = data["confirmed"]
    y = np.log(y)
    X = np.vstack((y, x))
    model = Model(exponential_logistic_for_plot)
    params = model.make_params(gamma=0, omega=0, alpha=0, beta_1=2)
    result = model.fit(z, X=X, params=params)
    best_params = result.params  # best fitted parameters
    y_pred = result.best_fit
    print(r2_score(z, y_pred))
    data["confirmed_pred"] = y_pred
    data["error"] = data["confirmed_pred"] - data["confirmed"]
    df = data[data["error"] < 0]
    x = df["day"]
    y = df["wuhan_outflow"]
    z = df["confirmed"]
    y = np.log(y)
    ax.scatter(x, y, z, s=50, marker=".", c="#DC143C")
    df = data[data["error"] >= 0]
    x = df["day"]
    y = df["wuhan_outflow"]
    z = df["confirmed"]
    y = np.log(y)
    ax.scatter(x, y, z, s=50, marker=".", c="k")
    _y = np.linspace(12, 15, 100)
    _x = np.linspace(1, days + 1, days)
    x, y = np.meshgrid(_x, _y)
    z = (best_params["alpha"] * np.exp(best_params["beta_1"] * y)) / (
            1 + np.exp(-1 * best_params["gamma"] * x + best_params["omega"]))
    ax.plot_surface(x, y, z, cmap="rainbow", alpha=0.65)
    ax.set_ylabel("Outflow population from Wuhan(log)", fontsize=14)
    ax.set_xlabel("Time", fontsize=14, rotation=-25)
    ax.set_zlim(zmin=0)
    plt.xticks(np.arange(1, days + 1, 2))
    ax.set_zlabel("Confirmed cases", fontsize=16, labelpad=10)
    plt.xlim(1, 27)
    plt.ylim(12, 15)
    plt.tight_layout()
    plt.savefig("nature-output/Figure-1(Hubei).png")
    plt.close()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15, azim=-35)
    data = panel[panel["province_en"] != "Hubei"]
    x = data["day"]
    y = data["wuhan_outflow"]
    z = data["confirmed"]
    y = np.log(y)
    X = np.vstack((y, x))
    model = Model(exponential_logistic_for_plot)
    params = model.make_params(gamma=0, omega=0, alpha=0, beta_1=2)
    result = model.fit(z, X=X, params=params)
    best_params = result.params  # best fitted parameters
    y_pred = result.best_fit
    data["confirmed_pred"] = y_pred
    print(r2_score(z, y_pred))
    data["error"] = data["confirmed_pred"] - data["confirmed"]
    df = data[data["error"] < 0]
    x = df["day"]
    y = df["wuhan_outflow"]
    z = df["confirmed"]
    y = np.log(y)
    ax.scatter(x, y, z, s=50, marker=".", c="#DC143C")
    df = data[data["error"] >= 0]
    x = df["day"]
    y = df["wuhan_outflow"]
    z = df["confirmed"]
    y = np.log(y)
    ax.scatter(x, y, z, s=50, marker=".", c="k")
    _y = np.linspace(0, 13, 50)
    _x = np.linspace(1, days + 1, days)
    x, y = np.meshgrid(_x, _y)
    z = (best_params["alpha"] * np.exp(best_params["beta_1"] * y)) / (
            1 + np.exp(-1 * best_params["gamma"] * x + best_params["omega"]))
    ax.plot_surface(x, y, z, cmap="rainbow", alpha=0.65)
    ax.set_ylabel("Outflow population from Wuhan(log)", fontsize=14)
    ax.set_xlabel("Time", fontsize=14, rotation=-25)
    ax.set_zlim(zmin=0)
    plt.xticks(np.arange(1, days + 1, 2))
    ax.set_zlabel("Confirmed cases", fontsize=16, labelpad=10)
    plt.xlim(1, 27)
    plt.ylim(0, 13)
    plt.tight_layout()
    plt.savefig("nature-output/Figure-1(without Hubei).png")


def plot_correlation_between_cases_and_outflow_log(x="wuhan_outflow", y="confirmed", size="population"):
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)
    for group in data.groupby(by=["date"]):
        title = group[0]
        df = group[1]
        plt.figure(figsize=(5, 5))
        plt.xscale("log")
        plt.yscale("log")
        coastal = [line.strip() for line in open("nature-data/coastal_cities.txt").readlines()]
        D = df[~df["city_en"].isin(
            ["Shuangyashan", "Hegang", "Jixi", "Qitaihe"])]  # Exclude some outlier data of Heilongjiang Province
        X = D[x].values + 1
        Y = D[y].values + 1
        lnX = np.log10(X)
        lnY = np.log10(Y)
        est = sm.OLS(lnY, sm.add_constant(lnX)).fit()
        a, b = est.params[0], est.params[1]
        X = np.arange(1, 10000000, 10000)
        Y = np.power(10, a + b * np.log10(X))
        plt.plot(X, Y, linestyle="--", c="gray")
        P = df[(df["province_en"] != "Hubei") & (~df["city_cn"].isin(coastal))]
        X = P[x].values + 1
        Y = P[y].values + 1
        z = P[size].values / 5 / 10000
        linewidth = 1
        edgecolor = "k"
        plt.scatter(X, Y, color="#4876FF", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        P = df[(df["province_en"] == "Hubei")]
        X = P[x].values + 1
        Y = P[y].values + 1
        z = P[size].values / 5 / 10000
        plt.scatter(X, Y, color="darkred", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        P = df[(df["city_cn"].isin(coastal))]
        X = P[x].values + 1
        Y = P[y].values + 1
        z = P[size].values / 5 / 10000
        plt.scatter(X, Y, color="orange", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        plt.scatter(0.1, 0.1, color="darkred", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Cities in Hubei province")
        plt.scatter(0.1, 0.1, color="orange", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Coastal cities")
        plt.scatter(0.1, 0.1, color="#4876FF", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Other citis")
        plt.scatter(3 * np.power(10, 5), 3, marker="o", s=700, c="None", linewidths=1, edgecolors="k")
        plt.scatter(1 * np.power(10, 6), 3, marker="o", s=300, c="None", linewidths=1, edgecolors="k")
        plt.scatter(2.5 * np.power(10, 6), 3, marker="o", s=150, c="None", linewidths=1, edgecolors="k")
        plt.scatter(5 * np.power(10, 6), 3, marker="o", s=80, c="None", linewidths=1, edgecolors="k")
        plt.text(2 * np.power(10, 3), 1.5, "Population size ------------------>")
        plt.title(title, fontsize=16)
        plt.ylim(1, 5000)
        plt.xlim(1, 10000000)
        plt.xlabel("Population outflow from Wuhan", fontsize=16)
        plt.ylabel("Confirmed cases", fontsize=16)
        plt.legend(loc=2, labelspacing=1, edgecolor="None", facecolor="None", markerscale=2, fontsize=12)
        plt.tight_layout()
        plt.savefig("nature-output/corr/%s(log).png" % title)
        plt.close()


def plot_correlation_between_cases_and_outflow_subgraph(x="wuhan_outflow", y="confirmed", size="population"):
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)
    ymax = 3000
    xmax = 2300000
    ymax2 = 200
    xmax2 = 100000
    for group in data.groupby(by=["date"]):
        title = group[0]
        df = group[1]
        fig = plt.figure(figsize=(5, 5))
        coastal = [line.strip() for line in open("nature-data/coastal_cities.txt").readlines()]
        D = df[~df["city_en"].isin(["Shuangyashan", "Hegang", "Jixi", "Qitaihe"])]
        X = D[x].values
        Y = D[y].values
        est = sm.OLS(Y, sm.add_constant(X)).fit()
        a, b = est.params[0], est.params[1]
        X = np.arange(1, 5000000, 100000)
        Y = a + b * X
        plt.plot(X, Y, linestyle="--", c="gray")
        P = df[(df["province_en"] != "Hubei") & (~df["city_cn"].isin(coastal))]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 10000
        linewidth = 1
        edgecolor = "k"
        plt.scatter(X, Y, color="#4876FF", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        P = df[(df["province_en"] == "Hubei")]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 10000
        plt.scatter(X, Y, color="darkred", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        P = df[(df["city_cn"].isin(coastal))]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 10000
        plt.scatter(X, Y, color="orange", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        plt.scatter(0.1, 0.1, color="darkred", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Cities in Hubei province")
        plt.scatter(0.1, 0.1, color="orange", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Coastal cities")
        plt.scatter(0.1, 0.1, color="#4876FF", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Other citis")

        plt.title(title, fontsize=16)
        plt.ylim(0, ymax)
        plt.xlim(0, xmax)
        plt.xticks([0, 1000000, 2000000],
                   [0, 1000000, 2000000])
        c1 = 0.78 * xmax
        plt.scatter(c1, 0.153 * ymax, marker="o", s=700, c="None", linewidths=1, edgecolors="k")
        plt.scatter(c1 + 0.08 * xmax, 0.153 * ymax, marker="o", s=300, c="None", linewidths=1, edgecolors="k")
        plt.scatter(c1 + 0.135 * xmax, 0.153 * ymax, marker="o", s=150, c="None", linewidths=1, edgecolors="k")
        plt.scatter(c1 + 0.175 * xmax, 0.153 * ymax, marker="o", s=80, c="None", linewidths=1, edgecolors="k")
        plt.text(c1 - 0.31 * xmax, 0.06 * ymax, "Population size ------------------>")
        plt.xlabel("Population outflow from Wuhan", fontsize=16)
        plt.ylabel("Confirmed cases", fontsize=16)
        plt.legend(loc=2, labelspacing=1, edgecolor="None", facecolor="None", markerscale=2, fontsize=12)
        left, bottom, width, height = 0.65, 0.35, 0.25, 0.25
        sub_ax = fig.add_axes([left, bottom, width, height])
        D = df[~df["city_en"].isin(["Shuangyashan", "Hegang", "Jixi", "Qitaihe"])]
        X = D[x].values
        Y = D[y].values
        est = sm.OLS(Y, sm.add_constant(X)).fit()
        a, b = est.params[0], est.params[1]
        X = np.arange(1, 5000000, 100000)
        Y = a + b * X
        # plt.plot(X, Y, linestyle="--", c="gray")
        P = df[(df["province_en"] != "Hubei") & (~df["city_cn"].isin(coastal))]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 30000
        linewidth = 1
        edgecolor = "k"
        plt.scatter(X, Y, color="#4876FF", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        P = df[(df["province_en"] == "Hubei")]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 30000
        plt.scatter(X, Y, color="darkred", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        P = df[(df["city_cn"].isin(coastal))]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 30000
        plt.scatter(X, Y, color="orange", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        plt.scatter(0.1, 0.1, color="darkred", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Cities in Hubei province")
        plt.scatter(0.1, 0.1, color="orange", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Coastal cities")
        plt.scatter(0.1, 0.1, color="#4876FF", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Other citis")
        plt.ylim(0, ymax2)
        plt.xlim(0, xmax2)
        plt.tight_layout()
        plt.savefig("nature-output/corr/%s(sub).png" % title)
        plt.close()


def plot_correlation_between_cases_and_outflow(x="wuhan_outflow", y="confirmed", size="population"):
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)
    ymax = 150
    xmax = 2300000
    for group in data.groupby(by=["date"]):
        title = group[0]
        df = group[1]
        plt.figure(figsize=(5, 5))
        coastal = [line.strip() for line in open("nature-data/coastal_cities.txt").readlines()]
        D = df[~df["city_en"].isin(["Shuangyashan", "Hegang", "Jixi", "Qitaihe"])]
        X = D[x].values
        Y = D[y].values
        est = sm.OLS(Y, sm.add_constant(X)).fit()
        a, b = est.params[0], est.params[1]
        X = np.arange(1, 5000000, 100000)
        Y = a + b * X
        plt.plot(X, Y, linestyle="--", c="gray")
        P = df[(df["province_en"] != "Hubei") & (~df["city_cn"].isin(coastal))]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 10000
        linewidth = 1
        edgecolor = "k"
        plt.scatter(X, Y, color="#4876FF", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        P = df[(df["province_en"] == "Hubei")]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 10000
        plt.scatter(X, Y, color="darkred", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        P = df[(df["city_cn"].isin(coastal))]
        X = P[x].values
        Y = P[y].values
        z = P[size].values / 5 / 10000
        plt.scatter(X, Y, color="orange", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor, s=z)
        plt.scatter(0.1, 0.1, color="darkred", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Cities in Hubei province")
        plt.scatter(0.1, 0.1, color="orange", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Coastal cities")
        plt.scatter(0.1, 0.1, color="#4876FF", marker="o", alpha=0.8, linewidths=linewidth, edgecolor=edgecolor,
                    label="Other citis")

        plt.title(title, fontsize=16)
        plt.ylim(0, ymax)
        plt.xlim(0, xmax)
        plt.xticks([0, 1000000, 2000000],
                   [0, 1000000, 2000000])
        c1 = 0.78 * xmax
        plt.scatter(c1, 0.153 * ymax, marker="o", s=700, c="None", linewidths=1, edgecolors="k")
        plt.scatter(c1 + 0.08 * xmax, 0.153 * ymax, marker="o", s=300, c="None", linewidths=1, edgecolors="k")
        plt.scatter(c1 + 0.135 * xmax, 0.153 * ymax, marker="o", s=150, c="None", linewidths=1, edgecolors="k")
        plt.scatter(c1 + 0.175 * xmax, 0.153 * ymax, marker="o", s=80, c="None", linewidths=1, edgecolors="k")
        plt.text(c1 - 0.31 * xmax, 0.06 * ymax, "Population size ------------------>")
        plt.xlabel("Population outflow from Wuhan", fontsize=16)
        plt.ylabel("Confirmed cases", fontsize=16)
        plt.legend(loc=2, labelspacing=1, edgecolor="None", facecolor="None", markerscale=2, fontsize=12)
        plt.tight_layout()
        plt.savefig("nature-output/corr/%s.png" % title)
        plt.close()


def plot_prediction_value(end_day=27, prediction=True):
    data = nature_model_submit.exponential_dynamic_model_estimate(end_day=end_day)
    data["actual-pred"] = data["confirmed"] - data["confirmed_pred"]
    D = data[data["day"] == end_day]
    D = D[["city_cn", "confirmed", "population", "wuhan_outflow"]]
    df = pd.DataFrame(data.groupby(by=["city_cn", "city_en", "province"], as_index=False)["actual-pred"].sum())
    df["risk_index"] = (df["actual-pred"] - df["actual-pred"].mean()) / df["actual-pred"].std()
    df = df.sort_values(by="risk_index", ascending=False)
    df = pd.merge(df, D, on="city_cn", how="left")
    df.to_excel("nature-output/risk_index_%d.xlsx" % end_day, index=False, encoding="utf-8-sig")
    df.set_index(df["city_cn"], inplace=True)
    x = df["risk_index"]
    mean, std = x.mean(), x.std(ddof=1)
    _x = np.arange(-10, 10, 1)
    conf_intveral = stats.norm.interval(0.90, loc=mean, scale=std)
    t90 = conf_intveral[1]
    l90 = conf_intveral[0]
    print(t90, l90)
    sns.distplot(x, bins=150, kde=False, fit=stats.norm, color="blue", rug=True)
    plt.axvline(x=t90, c="r")
    plt.axvline(x=l90, c="r")
    plt.xlim(-10, 10)
    plt.ylim(0, 2)
    plt.xlabel("Risk index $\overline{\Delta}_i$", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.tight_layout()
    plt.savefig("nature-output/risk_index_distribution_%d.png" % end_day)
    if prediction:
        x_ticks = ["Jan 24", "Jan 25", "Jan 26", "Jan 27", "Jan 28", "Jan 29", "Jan 30", "Jan 31", "Feb 1", "Feb 2",
                   "Feb 3", "Feb 4", "Feb 5", "Feb 6", "Feb 7", "Feb 8", "Feb 9", "Feb 10", "Feb 11", "Feb 12",
                   "Feb 13", "Feb 14", "Feb 15", "Feb 16", "Feb 17", "Feb 18", "Feb 19"]
        for group in data.groupby(by=["city_cn", "city_en"]):
            city_cn = group[0][0]
            city_en = group[0][1]
            plt.figure(figsize=(5, 5))
            risk_index = df["risk_index"][city_cn]
            y = group[1]["confirmed"]
            y_pred = group[1]["confirmed_pred"]
            x = group[1]["day"]
            plt.scatter(x, y, marker="^", color="r", label="Actual")
            plt.plot(x, y_pred, "-g", label="Prediction")
            plt.ylabel("Cases", fontsize=12)
            plt.ylim(ymin=0)
            plt.title(city_en, fontsize=12)
            plt.legend(edgecolor="None", fontsize=12)
            ymax = plt.axes().get_ylim()[1]
            plt.text(1, ymax * 3 / 4, "$\overline{\Delta}_i$=%.4f" % risk_index, fontsize=12)
            plt.xticks(x, x_ticks, fontsize=8, rotation=45)
            plt.tight_layout()
            plt.savefig("nature-output/predictions/%s.png" % city_cn)


def plot_increased_prediction_values(end_day=27):
    data = nature_model_submit.exponential_dynamic_increased_model_estimate(end_day=end_day)
    x_ticks = ["Jan l24", "Jan 25", "Jan 26", "Jan 27", "Jan 28", "Jan 29", "Jan 30", "Jan 31", "Feb 1", "Feb 2",
               "Feb 3", "Feb 4", "Feb 5", "Feb 6", "Feb 7", "Feb 8", "Feb 9", "Feb 10", "Feb 11", "Feb 12",
               "Feb 13", "Feb 14", "Feb 15", "Feb 16", "Feb 17", "Feb 18", "Feb 19", "Feb 20"]
    for group in data.groupby(by=["city_cn", "city_en"]):
        city_cn = group[0][0]
        city_en = group[0][1]
        plt.figure(figsize=(5, 5))
        y = group[1]["confirmed"]
        y_pred = group[1]["confirmed_pred"]
        x = group[1]["day"]
        plt.scatter(x, y, marker="^", color="r", label="Actual")
        plt.plot(x, y_pred, "-g", label="Prediction")
        plt.ylabel("Cases", fontsize=12)
        plt.ylim(ymin=0)
        plt.title(city_en, fontsize=12)
        plt.legend(edgecolor="None", fontsize=12)
        plt.xticks(x, x_ticks, fontsize=8, rotation=45)
        plt.tight_layout()
        plt.savefig("nature-output/increased/%s.png" % city_cn)


if __name__ == "__main__":
    # plot_dynamic_model_performance_3d()
    # plot_correlation_between_cases_and_outflow_log()
    # plot_correlation_between_cases_and_outflow_subgraph()
    # plot_correlation_between_cases_and_outflow()
    plot_prediction_value()
    plot_increased_prediction_values()
