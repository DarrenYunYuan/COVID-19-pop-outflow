import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats
import numpy as np
from lmfit import Model  # non-linear optimization by Levenberg-Marquardt algorithm

'''
城市统计年鉴（2018）中有298个城市，其中海南省三沙市比较特殊，没有人口数据，所以剔除；
剩余297个城市中缺失GDP（地区生产总值）数据的城市有：嘉峪关、儋州、中山、东莞和重庆
查询"中国统计信息网"相关报告补充这5个城市的GDP数据（2017年内），
1）嘉峪关（2109900万元）：http://www.tjcn.org/tjgb/28gs/35595.html
2）儋州（2880400万元）：http://www.tjcn.org/tjgb/21hn/35463.html
3）中山（34503100万元）：http://www.tjcn.org/tjgb/19gd/35456.html
4）东莞（75821200万元）：http://www.tjcn.org/tjgb/19gd/35455.html
5）重庆（195002700万元）：http://www.tjcn.org/tjgb/22cq/35464.html

注：疫情数据Wuhan-2019-nCoV.csv为病例实际统计时间，发布时间与统计相差1天（从2月7日起，直辖市除外）
疫情数据来源：https://github.com/canghailan/Wuhan-2019-nCoV
'''
'''
estimate method:
We use the source code library of lmfit in python
“LMFIT: Non-linear least-square minimization and curve-fitting for Python” (Newville et al. 2016)
to estimate the parameters in our models. The relevant codes are provided as follows.
'''
data_path = "nature-data/pneumonia_panel_296_cities(submit).csv"
days = 27

INF = 99999999999999
NAN = 0


def pearson_correlation_coefficient(y="confirmed", x=["Wuhan_outflow", "population", "GDP", "distance_to_wuhan(km)",
                                                      "novel_coronavirus_search", "sars_search",
                                                      "wuhan_pneumonia_search", "flu_search",
                                                      "atypical_pneumonia_search", "surgical_mask_search",
                                                      "n95_search"]):
    file = open("nature-output/corr_coef(daily).csv", "w", encoding="utf-8-sig")
    file.write("with confirmed cases:\n")
    headers = ["day"]
    headers.extend(x)
    file.write(",".join(headers))
    file.write("\n")
    data = pd.read_csv(data_path)
    X = []
    for k in x:
        X.append(k.replace("search", "search_addup"))
    for day in range(1, days + 1, 1):
        df = data[data["day"] == day]
        txt = [day]
        for k in X:
            txt.append(round(df[y].corr(df[k]), 4))
        txt = [str(item) for item in txt]
        file.write(",".join(txt))
        file.write("\n")
    file.write("with daily confirmed cases:\n")
    headers = ["day"]
    headers.extend(x)
    file.write(",".join(headers))
    file.write("\n")
    data = pd.read_csv(data_path)
    for day in range(1, days + 1, 1):
        df = data[data["day"] == day]
        df["N"] = df[y] - df["%s_pre" % y]
        df["N"] = df["N"].map(negative_number_to_zero)
        txt = [day]
        for k in x:
            txt.append(round(df["N"].corr(df[k]), 4))
        txt = [str(item) for item in txt]
        file.write(",".join(txt))
        file.write("\n")
    df = data[data["day"] == 27]  # 2月19日的截面数据
    x.insert(0, y)
    df[x].corr().to_excel("nature-output/corr_matrix.xlsx", encoding="utf-8-sig")


def standardization(data, variables):
    for var in variables:
        try:
            ln_var = "%s(Ln)" % var
            mean_var = "%s(Ln_Mean)" % var
            std_var = "%s(Ln_Std)" % var
            data[ln_var] = np.log(data[var] + 1)
            group = data.groupby("day", as_index=False)[ln_var].mean()
            group.columns = ["day", mean_var]
            data = pd.merge(data, group, on="day", how="left")
            group = pd.DataFrame(data.groupby("day")[ln_var].std())
            group.columns = [std_var]
            group["day"] = group.index
            group.reset_index(drop=True, inplace=True)
            data = pd.merge(data, group, on="day", how="left")
        except Exception as e:
            print(e)
    return data


def normalization(data, variables):
    for var in variables:
        sum_var = "%s(Sum)" % var
        group = data.groupby("day", as_index=False)[var].mean()
        group.columns = ["day", sum_var]
        data = pd.merge(data, group, on="day", how="left")
    return data


def province_dummies(data):
    return np.array(data[["Shanghai", "Yunnan", "Neimenggu", "Beijing", "Jilin", "Sichuan", "Tianjin", "Ningxia",
                          "Anhui", "Shandong", "Shanxi", "Guangdong", "Guangxi", "Xinjiang", "Jiangsu", "JIangxi",
                          "Hebei", "Henan", "Zhejiang", "Hainan", "Hubei", "Hunan", "Gansu", "Fujian", "Xizang",
                          "Guizhou", "Liaoning", "Chongqing", "Shanxi_", "Qinghai", "Heilongjiang"]].values)


# Daily Exponential-Static Model
def exponential_model(X, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2, lambda_3, lambda_4,
                      lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12, lambda_13,
                      lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20, lambda_21, lambda_22,
                      lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28, lambda_29, lambda_30,
                      lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha
    for i in range(N):
        R = R * np.exp(betas[i] * X[i])
    R = R * fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


def exponential_static_model_estimate_everyday(y="confirmed", x=["Wuhan_outflow", "GDP", "population"]):
    data = pd.read_csv(data_path)
    province = pd.read_csv("nature-data/province_fix.csv")
    data = pd.merge(data, province, on="province", how="left")
    data.fillna(0, inplace=True)
    data = standardization(data, x)
    file = open("nature-output/static_exp_model.csv", "w", encoding="utf-8-sig")
    headers = ["t", "R2", "α"]
    headers.extend(x)
    headers.append("N")
    file.write(",".join(headers))
    file.write("\n")
    for day in np.arange(1, days + 1, 1):
        df = data[data["day"] == day]
        Y = df[y]
        X = []
        for k in x:
            X.append((df["%s(Ln)" % k] - df["%s(Ln_Mean)" % k]) / df["%s(Ln_Std)" % k])
        fix = province_dummies(df)
        X.append(fix.T)
        X = np.vstack(X)
        model = Model(exponential_model)
        params = model.make_params(alpha=1, beta_1=1, beta_2=0, beta_3=0, beta_4=0, beta_5=0, lambda_1=0, lambda_2=0,
                                   lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0, lambda_8=0, lambda_9=0,
                                   lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0, lambda_14=0, lambda_15=0,
                                   lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0, lambda_20=0, lambda_21=0,
                                   lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0, lambda_26=0, lambda_27=0,
                                   lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
        result = model.fit(Y, X=X, params=params)
        best_params = result.params
        Y_pred = result.best_fit
        r2 = r2_score(Y, Y_pred)
        betas = ["beta_1", "beta_2", "beta_3", "beta_4", "beta_5"]
        line = [day, r2, best_params["alpha"]]
        for i in range(len(x)):
            line.append(round(best_params[betas[i]].value, 4))
        line = [str(i) for i in line]
        file.write(",".join(line))
        file.write("\n")
    file.close()


def exponential_static_model_estimate(date="2020-01-28", y="confirmed", x=["Wuhan_outflow", "GDP", "population"]):
    data = pd.read_csv(data_path)
    province = pd.read_csv("nature-data/province_fix.csv")
    data = pd.merge(data, province, on="province", how="left")
    data.fillna(0, inplace=True)
    data = standardization(data, x)
    data = data[data["date"] == date]
    Y = data[y]
    X = []
    for k in x:
        X.append((data["%s(Ln)" % k] - data["%s(Ln_Mean)" % k]) / data["%s(Ln_Std)" % k])
    fix = province_dummies(data)
    X.append(fix.T)
    X = np.vstack(X)
    model = Model(exponential_model)
    print(X)
    params = model.make_params(alpha=0, beta_1=0, beta_2=0, beta_3=0, beta_4=0, beta_5=0, lambda_1=0, lambda_2=0,
                               lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0, lambda_8=0, lambda_9=0,
                               lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0, lambda_14=0, lambda_15=0,
                               lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0, lambda_20=0, lambda_21=0,
                               lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0, lambda_26=0, lambda_27=0,
                               lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    result = model.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    data["confirmed_pred"] = Y_pred
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of Exponential model is ", r2)
    data["actual-pred"] = data[y] - data["%s_pred" % y]
    data["risk_index"] = (data["actual-pred"] - data["actual-pred"].mean()) / data["actual-pred"].std()
    x = data["actual-pred"]
    mean, std = x.mean(), x.std(ddof=1)
    conf_intveral = stats.norm.interval(0.9, loc=mean, scale=std)
    print(conf_intveral[1], conf_intveral[0])
    data = data.sort_values("actual-pred", ascending=False)
    data = data[["city_cn", "city_en", "province", y, "%s_pred" % y, "actual-pred", "risk_index"]]
    data.to_csv("nature-output/exponential_prediction_%s.csv" % date, encoding="utf-8-sig", index=False)


# Exponential-Logistic Dynamic Model
def exponential_logistic_model(X, gamma, omega, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2,
                               lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
                               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18,
                               lambda_19, lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26,
                               lambda_27, lambda_28, lambda_29, lambda_30, lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha / (1 + np.exp(-1 * gamma * X[0] + omega))
    for i in range(1, N):
        R *= np.exp(betas[i - 1] * X[i])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


# Exponential-Gompertz Dynamic Model
def exponential_gompertz_model(X, a, b, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2, lambda_3,
                               lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11,
                               lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
                               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27,
                               lambda_28, lambda_29, lambda_30, lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha * np.power(a, np.power(b, X[0]))
    for i in range(1, N):
        R *= np.exp(betas[i - 1] * X[i])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


# Exponential-Richards Dynamic Model
def exponential_richards_model(X, g, r, ti, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2,
                               lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
                               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18,
                               lambda_19, lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26,
                               lambda_27, lambda_28, lambda_29, lambda_30, lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha / np.power((1 + g * np.exp(-r * (X[0] - ti))), 1 / g)
    for i in range(1, N):
        R *= np.exp(betas[i - 1] * X[i])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


def exponential_dynamic_model_estimate(end_day=27, y="confirmed", x=["Wuhan_outflow", "GDP", "population"]):
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)
    data = standardization(data, x)
    data = data[data["day"] <= end_day]
    prov = pd.read_csv("nature-data/province_fix.csv")
    data = pd.merge(data, prov, on="province", how="left")
    data.fillna(0, inplace=True)
    Y = data[y]
    T = data["day"]
    X = [T]
    for k in x:
        X.append((data["%s(Ln)" % k] - data["%s(Ln_Mean)" % k]) / data["%s(Ln_Std)" % k])
    fix = province_dummies(data)
    X.append(fix.T)
    X = np.vstack(X)
    # fit the EL model
    ELmodel = Model(exponential_logistic_model)
    params = ELmodel.make_params(gamma=0.5, omega=1, alpha=1, beta_1=0, beta_2=0, beta_3=0, beta_4=0, beta_5=0,
                                 lambda_1=0, lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0,
                                 lambda_8=0, lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0,
                                 lambda_14=0, lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0,
                                 lambda_20=0, lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0,
                                 lambda_26=0, lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    result = ELmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of EL model is ", r2)
    data["confirmed_pred"] = Y_pred
    # fit the EG model
    EGmodel = Model(exponential_gompertz_model)
    params = EGmodel.make_params(a=0.5, b=0.5, alpha=1, beta_1=0, beta_2=0, beta_3=0, beta_4=0, beta_5=0, lambda_1=0,
                                 lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0, lambda_8=0,
                                 lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0, lambda_14=0,
                                 lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0, lambda_20=0,
                                 lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0, lambda_26=0,
                                 lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    params["a"].set(min=0, max=1)
    params["b"].set(min=0, max=1)
    result = EGmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of EG model is ", r2)
    # fit the ER model
    ERmodel = Model(exponential_richards_model)
    params = ERmodel.make_params(g=0.5, r=1, ti=1, alpha=1, beta_1=0, beta_2=0, beta_3=0, beta_4=0, beta_5=0,
                                 lambda_1=0, lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0,
                                 lambda_8=0, lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0,
                                 lambda_14=0, lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0,
                                 lambda_20=0, lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0,
                                 lambda_26=0, lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    result = ERmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of ER model is ", r2)
    return data[
        ["city_cn", "city_en", "province", "Wuhan_outflow", "GDP", "population", "day", "date", y, "%s_pred" % y]]


# Daily Power-Static Model
def power_model(X, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5,
                lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12, lambda_13, lambda_14,
                lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20, lambda_21, lambda_22, lambda_23,
                lambda_24, lambda_25, lambda_26, lambda_27, lambda_28, lambda_29, lambda_30, lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha
    for i in range(N):
        R *= np.power(X[i], betas[i])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


def power_static_model_estimate_everyday(y="confirmed", x=["Wuhan_outflow", "GDP", "population"]):
    data = pd.read_csv(data_path)
    province = pd.read_csv("nature-data/province_fix.csv")
    data = pd.merge(data, province, on="province", how="left")
    data.fillna(0, inplace=True)
    data = normalization(data, x)
    file = open("nature-output/static_power_model.csv", "w", encoding="utf-8-sig")
    headers = ["t", "R2", "α"]
    headers.extend(x)
    headers.append("N")
    file.write(",".join(headers))
    file.write("\n")
    for day in np.arange(1, days + 1, 1):
        df = data[data["day"] == day]
        Y = df[y]
        X = []
        for k in x:
            X.append(df[k] / df["%s(Sum)" % k])
        fix = province_dummies(df)
        X.append(fix.T)
        X = np.vstack(X)
        model = Model(power_model)
        params = model.make_params(alpha=1, beta_1=0.5, beta_2=0.5, beta_3=0.5, beta_4=0.5, beta_5=0.5, lambda_1=0,
                                   lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0, lambda_8=0,
                                   lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0, lambda_14=0,
                                   lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0, lambda_20=0,
                                   lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0, lambda_26=0,
                                   lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
        result = model.fit(Y, X=X, params=params)
        best_params = result.params  # best fitted parameters
        Y_pred = result.best_fit
        r2 = r2_score(Y, Y_pred)
        betas = ["beta_1", "beta_2", "beta_3", "beta_4", "beta_5"]
        line = [day, r2, best_params["alpha"]]
        for i in range(len(x)):
            line.append(round(best_params[betas[i]].value, 4))
        line = [str(i) for i in line]
        file.write(",".join(line))
        file.write("\n")
    file.close()


# Power-Logistic Dynamic Model
def power_logistic_model(X, gamma, omega, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2, lambda_3,
                         lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12,
                         lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20,
                         lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
                         lambda_29, lambda_30, lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha / (1 + np.exp(-1 * gamma * X[0] + omega))
    for i in range(1, N):
        R *= np.power(X[i], betas[i - 1])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


# Power-Gompertz Dynamic Model
def power_gompertz_model(X, a, b, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2, lambda_3, lambda_4,
                         lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12, lambda_13,
                         lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20, lambda_21,
                         lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28, lambda_29,
                         lambda_30, lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha * np.power(a, np.power(b, X[0]))
    for i in range(1, N):
        R *= np.power(X[i], betas[i - 1])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


# Power-Richards Dynamic Model
def power_richards_model(X, g, r, ti, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2, lambda_3,
                         lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12,
                         lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20,
                         lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
                         lambda_29, lambda_30, lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha / np.power((1 + g * np.exp(-r * (X[0] - ti))), 1 / g)
    for i in range(1, N):
        R *= np.power(X[i], betas[i - 1])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


def power_dynamic_model_estimate(end_day=27, y="confirmed", x=["Wuhan_outflow", "GDP", "population"]):
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)
    data = normalization(data, x)
    data = data[data["day"] <= end_day]
    province = pd.read_csv("nature-data/province_fix.csv")
    data = pd.merge(data, province, on="province", how="left")
    Y = data[y]
    T = data["day"]
    X = [T]
    for k in x:
        X.append(data[k] / data["%s(Sum)" % k])
    fix = province_dummies(data)
    X.append(fix.T)
    X = np.vstack(X)
    # fit the PL model
    PLmodel = Model(power_logistic_model)
    params = PLmodel.make_params(gamma=1, omega=1, alpha=0.5, beta_1=0.5, beta_2=0.5, beta_3=0.5, beta_4=0.5,
                                 beta_5=0.5, lambda_1=0, lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0,
                                 lambda_7=0, lambda_8=0, lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0,
                                 lambda_14=0, lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0,
                                 lambda_20=0, lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0,
                                 lambda_26=0, lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    result = PLmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of PL model is ", r2)
    # fit the PG model
    PGmodel = Model(power_gompertz_model)
    params = PGmodel.make_params(a=0.5, b=0.5, alpha=0.5, beta_1=0.5, beta_2=0.5, beta_3=0.5, beta_4=0.5, beta_5=0.5,
                                 lambda_1=0, lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0,
                                 lambda_8=0, lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0,
                                 lambda_14=0, lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0,
                                 lambda_20=0, lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0,
                                 lambda_26=0, lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    params["a"].set(min=0, max=1)
    params["b"].set(min=0, max=1)
    result = PGmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of PG model is ", r2)
    # fit the PR model
    PRmodel = Model(power_richards_model)
    params = PRmodel.make_params(g=0.5, r=1, ti=1, alpha=0.5, beta_1=0.5, beta_2=0.5, beta_3=0.5, beta_4=0.5,
                                 beta_5=0.5, lambda_1=0, lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0,
                                 lambda_7=0, lambda_8=0, lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0,
                                 lambda_14=0, lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0,
                                 lambda_20=0, lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0,
                                 lambda_26=0, lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    result = PRmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of PR model is ", r2)


# Exponential-Logistic Dynamic Increased Model
def exponential_logistic_increased_model(X, gamma, omega, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1,
                                         lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9,
                                         lambda_10, lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16,
                                         lambda_17, lambda_18, lambda_19, lambda_20, lambda_21, lambda_22, lambda_23,
                                         lambda_24, lambda_25, lambda_26, lambda_27, lambda_28, lambda_29, lambda_30,
                                         lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha * np.exp(-1 * gamma * X[0] + omega) * gamma / np.power((1 + np.exp(-1 * gamma * X[0] + omega)), 2)
    for i in range(1, N):
        R *= np.exp(betas[i - 1] * X[i])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


# Exponential-Gompertz Dynamic Increased Model
def exponential_gompertz_increased_model(X, a, b, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2,
                                         lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9,
                                         lambda_10, lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16,
                                         lambda_17, lambda_18, lambda_19, lambda_20, lambda_21, lambda_22, lambda_23,
                                         lambda_24, lambda_25, lambda_26, lambda_27, lambda_28, lambda_29, lambda_30,
                                         lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha * np.power(a, np.power(b, X[0])) * np.power(b, X[0]) * np.log(a) * np.log(b)
    for i in range(1, N):
        R *= np.exp(betas[i - 1] * X[i])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


def negative_number_to_zero(x):
    if x < 0:
        return 0
    else:
        return x


def exponential_dynamic_increased_model_estimate(end_day=27, y="confirmed", x=["Wuhan_outflow", "GDP", "population"]):
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)
    data = standardization(data, x)
    data = data[data["day"] <= end_day]
    prov = pd.read_csv("nature-data/province_fix.csv")
    data = pd.merge(data, prov, on="province", how="left")
    data.fillna(0, inplace=True)
    data[y] = data[y] - data["%s_pre" % y]
    data[y] = data[y].map(negative_number_to_zero)
    Y = data[y]
    T = data["day"]
    X = [T]
    for k in x:
        X.append((data["%s(Ln)" % k] - data["%s(Ln_Mean)" % k]) / data["%s(Ln_Std)" % k])
    fix = province_dummies(data)
    X.append(fix.T)
    X = np.vstack(X)
    # fit the EL model
    ELmodel = Model(exponential_logistic_increased_model)
    params = ELmodel.make_params(gamma=0.5, omega=1, alpha=1, beta_1=0, beta_2=0, beta_3=0, beta_4=0, beta_5=0,
                                 lambda_1=0, lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0,
                                 lambda_8=0, lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0,
                                 lambda_14=0, lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0,
                                 lambda_20=0, lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0,
                                 lambda_26=0, lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    result = ELmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of EL model is ", r2)
    data["confirmed_pred"] = Y_pred
    # fit the EG model
    EGmodel = Model(exponential_gompertz_increased_model)
    params = EGmodel.make_params(a=0.5, b=0.5, alpha=1, beta_1=0, beta_2=0, beta_3=0, beta_4=0, beta_5=0, lambda_1=0,
                                 lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0, lambda_8=0,
                                 lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0, lambda_14=0,
                                 lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0, lambda_20=0,
                                 lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0, lambda_26=0,
                                 lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    params["a"].set(min=0, max=1)
    params["b"].set(min=0, max=1)
    result = EGmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of EG model is ", r2)
    return data[["city_cn", "city_en", "province", "day", "date", y, "%s_pred" % y]]


# Power-Logistic Dynamic Increased Model
def power_logistic_increased_model(X, gamma, omega, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1,
                                   lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9,
                                   lambda_10, lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16,
                                   lambda_17, lambda_18, lambda_19, lambda_20, lambda_21, lambda_22, lambda_23,
                                   lambda_24, lambda_25, lambda_26, lambda_27, lambda_28, lambda_29, lambda_30,
                                   lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha * np.exp(-1 * gamma * X[0] + omega) * gamma / np.power((1 + np.exp(-1 * gamma * X[0] + omega)), 2)
    for i in range(1, N):
        R *= np.power(X[i], betas[i - 1])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


# Power-Gompertz Dynamic Increased Model
def power_gompertz_increased_model(X, a, b, alpha, beta_1, beta_2, beta_3, beta_4, beta_5, lambda_1, lambda_2, lambda_3,
                                   lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11,
                                   lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18,
                                   lambda_19, lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25,
                                   lambda_26, lambda_27, lambda_28, lambda_29, lambda_30, lambda_31):
    lambdas = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
               lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19,
               lambda_20, lambda_21, lambda_22, lambda_23, lambda_24, lambda_25, lambda_26, lambda_27, lambda_28,
               lambda_29, lambda_30, lambda_31]
    betas = [beta_1, beta_2, beta_3, beta_4, beta_5]
    N = len(X) - 31
    fix = 1
    for i in range(len(lambdas)):
        fix = fix * np.exp(lambdas[i] * X[i + N])
    R = alpha * np.power(a, np.power(b, X[0])) * np.power(b, X[0]) * np.log(a) * np.log(b)
    for i in range(1, N):
        R *= np.power(X[i], betas[i - 1])
    R *= fix
    R[np.isinf(R)] = INF
    R[np.isnan(R)] = NAN
    return R


def power_dynamic_increased_model_estimate(end_day=27, y="confirmed", x=["Wuhan_outflow", "GDP", "population"]):
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)
    data = normalization(data, x)
    data = data[data["day"] <= end_day]
    prov = pd.read_csv("nature-data/province_fix.csv")
    data = pd.merge(data, prov, on="province", how="left")
    data.fillna(0, inplace=True)
    data[y] = data[y] - data["%s_pre" % y]
    data[y] = data[y].map(negative_number_to_zero)
    Y = data[y]
    T = data["day"]
    X = [T]
    for k in x:
        X.append(data[k] / data["%s(Sum)" % k])
    fix = province_dummies(data)
    X.append(fix.T)
    X = np.vstack(X)
    # fit the PL model
    PLmodel = Model(power_logistic_increased_model)
    params = PLmodel.make_params(gamma=0.5, omega=1, alpha=1, beta_1=0.5, beta_2=0.5, beta_3=0.5, beta_4=0.5,
                                 beta_5=0.5, lambda_1=0, lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0,
                                 lambda_7=0, lambda_8=0, lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0,
                                 lambda_14=0, lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0,
                                 lambda_20=0, lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0,
                                 lambda_26=0, lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    result = PLmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of PL model is ", r2)
    data["confirmed_pred"] = Y_pred
    # fit the EG model
    PGmodel = Model(power_gompertz_increased_model)
    params = PGmodel.make_params(a=0.5, b=0.5, alpha=1, beta_1=0.5, beta_2=0.5, beta_3=0.5, beta_4=0.5, beta_5=0.5,
                                 lambda_1=0,
                                 lambda_2=0, lambda_3=0, lambda_4=0, lambda_5=0, lambda_6=0, lambda_7=0, lambda_8=0,
                                 lambda_9=0, lambda_10=0, lambda_11=0, lambda_12=0, lambda_13=0, lambda_14=0,
                                 lambda_15=0, lambda_16=0, lambda_17=0, lambda_18=0, lambda_19=0, lambda_20=0,
                                 lambda_21=0, lambda_22=0, lambda_23=0, lambda_24=0, lambda_25=0, lambda_26=0,
                                 lambda_27=0, lambda_28=0, lambda_29=0, lambda_30=0, lambda_31=0)
    params["a"].set(min=0, max=1)
    params["b"].set(min=0, max=1)
    result = PGmodel.fit(Y, X=X, params=params)
    Y_pred = result.best_fit
    r2 = r2_score(Y, Y_pred)
    print(result.fit_report())
    print("R square of EL model is ", r2)
    return data[["city_cn", "city_en", "province", "day", "date", y, "%s_pred" % y]]


if __name__ == "__main__":
    pearson_correlation_coefficient()
    # exponential_static_model_estimate_everyday()
    # exponential_static_model_estimate()
    # exponential_dynamic_model_estimate()
    # power_static_model_estimate_everyday()
    # power_dynamic_model_estimate()
