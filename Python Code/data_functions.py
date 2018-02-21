def annualize_quarterly(keys, data_dict, fqtr_dict):
    """
    Annualizes the quarterly EBITDA or Interest Expense data using the following rules:
    Q1_ann = Q1 * 4
    Q2_ann = (Q2 + Q1) * 2
    Q3_ann = (Q3 + Q2 + Q1) + ((Q3 + Q2 + Q1) / 3)
    Q4_ann = Q4 + Q3 + Q2 + Q1

    :param keys: The list containing all unique gvkeys for which we want to annualize data.
    :param data_dict: The dictionary containing our to be annualized data.
    :param fqtr_dict: The dictionary containing the quarters.
    :return: Annualized data per quarter.
    """

    data_ann_dict = {}

    for i in range(len(keys)):
        k = keys[i]
        fqtr = fqtr_dict[k].tolist()
        data = data_dict[k].tolist()
        annual_list = []

        for ii in range(len(data)):
            if fqtr[ii] == 1:
                annual = data[ii] * 4
                annual_list.append(annual)
            elif fqtr[ii] == 2 and fqtr[ii-1] == 1:
                annual = (data[ii] + data[ii - 1]) * 2
                annual_list.append(annual)
            elif fqtr[ii] == 2 and fqtr[ii-1]:
                annual = data[ii] * 4
                annual_list.append(annual)
            elif fqtr[ii] == 3 and fqtr[ii-1] == 2:
                summed = data[ii] + data[ii - 1] + data[ii - 2]
                annual = summed + (summed / 3)
                annual_list.append(annual)
            elif fqtr[ii] == 3:
                annual = data[ii] * 4
                annual_list.append(annual)
            elif fqtr[ii] == 4 and fqtr[ii-1] == 3 and fqtr[ii-2] == 2 and fqtr[ii-3] == 1:
                annual = data[ii] + data[ii - 1] + data[ii - 2] + data[ii - 3]
                annual_list.append(annual)
            else:
                annual = data[ii] * 4
                annual_list.append(annual)
            data_ann_dict.update({k: annual_list})

    return data_ann_dict


def annualize_qytd(keys, data_dict, fqtr_dict):
    """
    Annualize quarterly YTD data, for example capxytd, using the following rules:
    Q1_ann = Q1 * 4
    Q2_ann = Q2 * 2
    Q3_ann = Q3 + (Q3 / 3)
    Q4_ann = Q4

    :param keys: The list containing all unique gvkeys for which we want to annualize data.
    :param data_dict: The dictionary containing our to be annualized data.
    :param fqtr_dict: The dictionary containing the quarters.
    :return: Annualized data per quarter.
    """

    data_ann_dict = {}

    for i in range(len(keys)):
        k = keys[i]
        fqtr = fqtr_dict[k].tolist()
        data = data_dict[k].tolist()
        annual_list = []

        for ii in range(len(data)):
            if fqtr[ii] == 1:
                annual = data[ii] * 4
                annual_list.append(annual)
            elif fqtr[ii] == 2:
                annual = data[ii] * 2
                annual_list.append(annual)
            elif fqtr[ii] == 3:
                annual = data[ii] + (data[ii] / 3)
                annual_list.append(annual)
            else:
                annual_list.append(data[ii])

        data_ann_dict.update({k: annual_list})

    return data_ann_dict
