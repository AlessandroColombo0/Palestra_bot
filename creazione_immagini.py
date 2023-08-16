from funzioni import schedaFname_to_title, weekday_nday_month_year, ora_EU, md_v2_replace, stringaPeso_to_num

import copy
import math

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

import calendar
import datetime as dt

import pypdfium2 as pdfium

import numpy as np
from PIL import Image

import os
import xlsxwriter
import convertapi  # https://www.convertapi.com

from icecream import ic
ic.configureOutput(prefix="> ", includeContext=True)
# ic.disable()


nomi_in_scheda = {"pettobi": "Petto Bicipiti", "schiena": "Schiena", "spalletri": "Spalle Tricipiti", "altro": "Gambe", "addominali": "Addominali"}

def days_to_mesiSett(days):
    """ 30 -> 1, 0      45 -> 1, 2      60 -> 2, 0 """
    months = days//30
    remaining_days = days-months*30
    sett = remaining_days//7
    return months, sett



# PESO

def Grafico_peso(lista_peso_bilancia, num_months):
    ic(lista_peso_bilancia)
    class Date_num():
        """ questa classe è un numero che quando dopo una sottrazione va sotto allo 0 ripare da 12. es: 3 - 5 = 10.
        funziona anche per le addizioni """
        def __init__(self, month, year):
            self.month = month
            self.year = year

        def __add__(self, add_num):
            self.month = self.month + add_num
            cycles = self.month // 12

            self.month -= 12*cycles
            self.year += 1*cycles

        def __sub__(self, sub_num):
            self.month = self.month - sub_num
            cycles = 0

            full = 1
            while True:
                if self.month < full:
                    full -= 12
                    cycles += 1
                else:
                    break

            self.month += 12*cycles
            self.year -= 1*cycles


    dt_today = ora_EU()
    iso_today = dt_today.isoformat()
    month_today = int(iso_today[5:7])
    year_today = int(iso_today[0:4])

    num_days = num_months*30
    prima_data = dt_today - dt.timedelta(num_days)
    prima_data_idx = None

    # selezione intervallo da cui prendere i dati per il grafico finale
    for idx, ele in enumerate(lista_peso_bilancia):
        if dt.datetime.fromisoformat(ele[0]) > prima_data:  # se la data i è più avanti rispetto alla data da cui iniziamo a vedere il grafico
            prima_data_idx = idx
            break

    if prima_data_idx == None:
        return False

    else:

        def ora_to_num(date_string):
            """ num minuti in un orario, serve per la colormap e per i tick di essa"""
            hours, mins = int(date_string[11:13]), int(date_string[14:16])
            return hours*60 + mins

        bilancia_dtime = [dt.datetime.fromisoformat(i[0]) for i in lista_peso_bilancia[prima_data_idx:]]
        bilancia_peso = [i[1] for i in lista_peso_bilancia[prima_data_idx:]]
        n_ora = [ora_to_num(i[0]) for i in lista_peso_bilancia[prima_data_idx:]]

        max_peso = max(bilancia_peso)
        min_peso = min(bilancia_peso)
        diff_peso = max_peso-min_peso

        nOra_data_peso_dict = {"data": bilancia_dtime, "peso": bilancia_peso, "nOra": n_ora}


        plt.figure(figsize=(24,10))

        sns.set_style("darkgrid", {"grid.linestyle": ":", "grid.color": "0.82"})
        sns.set_context("notebook", rc={"grid.linewidth": 1.8})


        # LINEE PRIMO DEL MESE E NOME DEL MESE SUL GRAFICO

        mesi = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
        ultimo_g = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # lista contenente l'ultimo giorno di ogni mese

        today_date_num = Date_num(month_today, year_today)
        today_date_num + 1  # aggiungiamo 1 perchè nel for loop toglieremo subito 1, e se non aggiungessimo 1 partiremmo dal mese prima di quello in cui siamo

        for month_i in range(num_months+1):
            # LINEA CHE SEGNA L'1 DI OGNI MESE
            today_date_num - 1
            # today date num ci dà l'anno in cui siamo e il mese sottranedo 1, è compatibile su più anni, se siamo a dicembre (12) e facciamo - 1 otteniamo gennaio (1) e l'anno (2022)
            ## si abbassa di 1 (2021)
            list_month_index = today_date_num.month - 1  # questo è l'index che useremo per riferirci al mese nelle liste, questo perchè gennaio in today_date_num.month è 1,
            plt.axvline(dt.datetime(today_date_num.year, today_date_num.month, 1), c="0.6", linewidth=2, zorder=1)
            ## nelle liste dovrebbe essere 0 e così via per tutti i numeri (uno in meno)

            # TESTO DEL MESE
            text_x_coor = mdates.datestr2num((dt.datetime(today_date_num.year, today_date_num.month, ultimo_g[list_month_index])
                                              - dt.timedelta(days=ultimo_g[list_month_index]/2 - 1)).isoformat())  # sottraiamo i giorni che corrispondono alla metà dell'ultimo giorno
            ## del mese, in quesot modo  le coordinate del testo sono al centro

            plt.text(text_x_coor, max_peso-(diff_peso*0.3), mesi[list_month_index],
                     color="0.6", fontsize=36, verticalalignment="center", horizontalalignment="center", weight="bold", clip_on=True, zorder=1)  # clip on fa in modo che il testo sia
            #   nel plot, se non lo mettiamo le scritte di alcuni mesi si vederebbero al di fuori dello scatter plot


        # COLORMAP

        palette = sns.hls_palette(l=.4, as_cmap=True)
        # palette = sns.cubehelix_palette(n_colors=6, start=2, rot=0.4, gamma=1.0, hue=0.8, light=0.85, dark=0.15, reverse=False, as_cmap=True)
        ax = sns.scatterplot(data=nOra_data_peso_dict, x="data", y="peso", s=1550,  # s = size
                             hue="nOra", hue_norm=(0,24*60), palette=palette, alpha=0.9, linewidth=0,  # linewidth 0 toglie il bordo bianco dai pallini
                             zorder=3)  # se non lo mettiamo le righe sotto sono sopra ad esso

        ax.set_xlim([dt_today - dt.timedelta(days=num_months*30), dt_today])
        # ax.set_ylim([dt_today - dt.timedelta(days=num_months*30), dt_today])

        str_data_1 = f"{int(bilancia_dtime[0].isoformat()[8:10])} {mesi[int(bilancia_dtime[0].isoformat()[5:7]) -1]}"
        str_data_2 = f"{int(bilancia_dtime[-1].isoformat()[8:10])} {mesi[int(bilancia_dtime[-1].isoformat()[5:7]) -1]}"
        ax.set_xlabel(f"Data - dal {str_data_1} al {str_data_2}", fontsize=25)
        ax.set_ylabel(f"Peso (Kg) - da {min_peso}kg a {max_peso}kg", fontsize=25)


        # KG SUI PALLINI
        for data, peso in zip(nOra_data_peso_dict["data"], nOra_data_peso_dict["peso"]):
            text_x_coor = mdates.datestr2num(data.isoformat())  # sytiamo usando il plot come tempo quindi dobbiamo trasformare il tempo in un numero che esista come coordinata sul plot
            plt.text(text_x_coor, peso, peso, color="white", alpha=0.5, fontsize=12, verticalalignment="center", horizontalalignment="center", weight="bold", zorder=3)


        # COLORBAR

        norm = plt.Normalize(0, 60*24)
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        colorbar = ax.figure.colorbar(sm, ticks=[i*120 for i in range(13)])

        # ticks colorbar

        tick_labels = []
        for i in range(0, 26, 2):
            if i < 13:
                meridium = "AM"
            else:
                meridium = "PM"
                i -= 12

            tick_labels.append(f"{i}:00 {meridium}")

        colorbar.ax.set_yticklabels(tick_labels)  # horizontal colorbar


        # PLOT PESO MEDIO PER INTERVALLI

        rev_bilancia_dtime = copy.deepcopy(bilancia_dtime)
        rev_bilancia_dtime.reverse()  # date dalla più recente alla più vecchia
        # rev_bilancia_dtime.append("padding")  # appendiamo del padding, in questo modo non avremo errore (per aver finito gli elementi) quando saremo all'ultimo index

        days_delta = 7
        x_middleWeek_dates = []  # questo dizionario contiene le date in mezzo alle settimane, sarà l'x dei triangoli neri, parte dall'ultima data (più recente) fino ad arrivare alla
            # prima
        interval_list_idxs = []
        empty = True

        import math
        days_delta_list = [i*7 for i in range(1, math.ceil(num_days/days_delta) + 1)]  # num days / 7 arrotondato per difetto, in questo modo abbiamo il numero massimo possibile di
        # settimane
        intervals_dt_list = [dt_today - dt.timedelta(days_delta) for days_delta in days_delta_list]

        for inter_data in intervals_dt_list:
            idx_replace_list = []
            x_middleWeek_dates.append(inter_data + dt.timedelta(3.5))
            interval_list_idxs.append([])

            for idx, data in enumerate(rev_bilancia_dtime):
                if data != "0":
                    if data > inter_data:
                        empty = False
                        interval_list_idxs[-1].append(idx)
                        idx_replace_list.append(idx)

                    else:
                        if empty == False:
                            empty = True
                            for idx in idx_replace_list:
                                rev_bilancia_dtime[idx] = "0"  # mettiamo 0, in questo modo possiamo riconoscere se una data è già stata processata
                            break
                        else:  # se non c'erano dati in quell'intervallo
                            interval_list_idxs[-1].append("empty")
                            break


        rev_bilancia_peso = copy.deepcopy(bilancia_peso)
        ic(bilancia_peso)
        rev_bilancia_peso.reverse()

        # peso medio per intervallo
        mean_interval_weights = []
        for interval in interval_list_idxs:
            if interval[0] != "empty":
                mean_interval_weights.append(sum([rev_bilancia_peso[idx] for idx in interval]) / len(interval))
            else:
                mean_interval_weights.append("empty")

        empty_idxs = []
        pre_idxs = []
        after_idxs = []
        for i, idx in enumerate(mean_interval_weights):
            if idx == "empty":
                empty_idxs.append(i)
            else:
                # after
                if i != 0:
                    if mean_interval_weights[i-1] == "empty":
                        after_idxs.append(i)
                # pre
                if i != len(mean_interval_weights) - 1:
                    if mean_interval_weights[i+1] == "empty":
                        pre_idxs.append(i)

        empty_streak = [[]]
        streak_num = 0
        for i, idx in enumerate(empty_idxs):
            empty_streak[streak_num].append(idx)

            if i != len(empty_idxs) - 1:  # se non siamo all'ultima iterazione
                if empty_idxs[i+1] == idx + 1:
                    pass
                else:
                    empty_streak.append([])
                    streak_num += 1

        # EMPTY TO PESO
        # handling "empty" ad inizio e fine lista
        if empty_streak != [[]]:  # se ci sono settimane empty

            if empty_streak[0][0] == 0:  # se la prima settimana è empty
                for idx in empty_streak[0]:  # per ogni index di questo streak mettiamo come numero il primo numero disponibile
                    mean_interval_weights[idx] = mean_interval_weights[after_idxs[0]]
                del after_idxs[0]
                del empty_streak[0]

            if len(empty_streak) > 1:
                if empty_streak[-1][-1] == len(mean_interval_weights) - 1:
                    for idx in empty_streak[-1]:
                        mean_interval_weights[idx] = mean_interval_weights[pre_idxs[-1]]
                    del pre_idxs[-1]
                    del empty_streak[-1]

        # empty rimanenti
        empty_replacements = []
        for i, pre_idx in enumerate(pre_idxs):
            empty_replacements.append([])
            peso_1 = mean_interval_weights[pre_idx]
            peso_2 = mean_interval_weights[after_idxs[i]]

            length = after_idxs[i] - pre_idx

            peso_diff = peso_2 - peso_1
            kg_per_step = peso_diff/length

            for j in range(1, length):
                empty_replacements[-1].append(peso_1 + kg_per_step*j)

        for i, streak in enumerate(empty_streak):
            for j, idx in enumerate(streak):
                mean_interval_weights[idx] = empty_replacements[i][j]


        # PLOT

        # round_tri_marker = MarkerStyle('1', capstyle=CapStyle.round)

        marker_outer = dict(marker="1",
                            markersize=18,
                            markeredgecolor=(0, 0, 0, 0.8),
                            markeredgewidth=3)

        plt.plot(x_middleWeek_dates, mean_interval_weights,
                 **marker_outer, solid_capstyle="round",
                 color=(0, 0, 0, 0.5), zorder=1, linewidth=4)

        # INTERVALLO X TICKS
        x_middleWeek_dates_grid = [xmw-dt.timedelta(days=3.5) for xmw in x_middleWeek_dates]  # spostiamo di 3.5 in avanti, in questo modo ottenimo degli xticks che fanno
            # vedere gli intervalli in cui vengono fatte le medie del peso
        plt.xticks(x_middleWeek_dates_grid)


        # ADJUSTED LINE OF BEST FIT
        # in questa parte venogno create le linee di best fit aggiustate (usando la media ad intervalli) per i mesi selezionati e i mesi selezionati - i

        # lobf dei mesi selezionati

        def best_fit(x, y):
            n = len(x) # or len(y)
            x_mean = sum(x)/n
            y_mean = sum(y)/n

            numer = sum([xi*yi for xi, yi in zip(x, y)]) - x_mean*y_mean*n
            denum = sum([xi**2 for xi in x]) - x_mean**2*n

            m = numer / denum
            b = y_mean - m * x_mean

            return m, b

        x_mW_dates_toNum = [mdates.datestr2num(data.isoformat()) for data in x_middleWeek_dates]
        m, b = best_fit(x_mW_dates_toNum, mean_interval_weights)
        y_lobf = [(x_mW_dates_toNum[0]+10)*m+b, (x_mW_dates_toNum[-1]-10)*m+b]  # moltiplichiamo m e aggiungiamo b ai numeri delle date, in questo modo otteniamo le coordinate y per
            # quei punti
        # + 10 e - 10 ci permettono di avere una linea che sembra proseguire all'infinito

        # colore
        # ci basiamo sul fatto che guadagnare 1kg al mese ha il colore più intenso

        y1 = x_mW_dates_toNum[0]*m+b  # x_mW_dates_toNum[0]: data più recente
        y2 = x_mW_dates_toNum[-1]*m+b  # x_mW_dates_toNum[-1]: data più vecchia

        def alpha_calculator(y1, y2):
            diff = y1 - y2  # kg guadagnati / persi nel periodo basandosi sulla line of best fit

            segno = ""
            if diff > 0:
                segno = "+"
                color = "limegreen"
            else:
                color = "crimson"
            scaled_diff = diff / num_months  # scaliamo in base al periodo e otteniamo il valore assoluto
            alpha = abs(scaled_diff)
            alpha += 0.20  # aggiungiamo 0.20 all'alpha così da non ottenere delle linee molto trasparenti in alcuni casi
            if alpha > 1:
                alpha = 1

            return alpha, color, segno, diff, scaled_diff

        alpha, color, segno, diff, scaled_diff = alpha_calculator(y1, y2)

        plt.plot([x_mW_dates_toNum[0]+10, x_mW_dates_toNum[-1]-10], y_lobf,  # riportiamo +-10 anche qua in modo che le coordinate x corrispondano alle y
                 zorder=2, alpha=alpha, color=color, linewidth=5)

        # lobf dei mesi i più vicini

        _30Days_indexes = [[] for _ in range(num_months-1)]  # se abbiamo selezionato 3 mesi: [[], []], conterrà gli index di x_middleWeek_dates dei primi 30 giorni a [0],
            # quelli da 30 a 60 in [1] e così via
        cycle = 0
        for i, date in enumerate(x_middleWeek_dates):
            if date > dt_today - dt.timedelta((cycle+1)*30):  # se la data i è più grande di oggi - (30 giorni * ciclo)
                _30Days_indexes[cycle].append(i)
            else:
                if cycle == num_months-2:  # es: se num_months = 3 e cycle = 1 dobbiamo smettere perchè cycle = 2 andrebbe nel terzo mese prima
                    break
                cycle += 1
                _30Days_indexes[cycle].append(i)

        lastIdxs_30days = [idxs[-1] for idxs in _30Days_indexes]

        """
        ES:
         _30Days_indexes: [[0, 1, 2, 3],                                # mese più recente
                           [0, 1, 2, 3, 4, 5, 6, 7, 8],                 # ultimi 2 mesi
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]  # ultimi 3 mesi
        """

        for i, last_idx in enumerate(lastIdxs_30days):
            interval_weights = [weight for weight in mean_interval_weights[:last_idx]]
            interval_dates = [date for date in x_mW_dates_toNum[:last_idx]]
            m, b = best_fit(interval_dates, interval_weights)

            y1, y2 = x_mW_dates_toNum[0]*m+b, x_mW_dates_toNum[last_idx]*m+b
            alpha, color, _, _, _ = alpha_calculator(y1, y2)

            plt.plot([x_mW_dates_toNum[0], x_mW_dates_toNum[last_idx]], [y1, y2],
                     zorder=2, alpha=alpha, color=color, linewidth=2.5, linestyle="--")

        fname = "immagini/grafico_peso.png"
        plt.savefig(fname, bbox_inches='tight', dpi=300)


        # CAPTION

        msg = f"""<b>GRAFICO DEL PESO</b>
        Periodo: <code>{num_months}</code> mesi
        Totale pesi: <code>{len(rev_bilancia_peso)}</code>
              
        <b>Line of best fit</b>:
        da <code>{round(y1, 2)}㎏</code> a <code>{round(y2, 2)}㎏</code> (<code>{segno}{round(diff, 2)}㎏</code>)
        <code>{segno}{round(scaled_diff, 2)}kg</code> al mese
        """

        return fname, msg



# 4 SETT

def Grafico_4sett(db, data_prossima_stat, F_somma_numeri_in_stringa, D_m_to_mesi):
    # try:
    diz_sett = {"sett_0": [], "sett_1": [], "sett_2": [], "sett_3": [], "sett_4": []}

    for allenamento in db["allenamenti"]:
        # allenamento =
        """
        [
            "2021-05-01T00:00:00",
            "pettobi"
        ],
        """
        data_allenamento = dt.datetime.fromisoformat(allenamento[0])

        if (data_prossima_stat - data_allenamento).days <= 34:
            if (data_prossima_stat - data_allenamento).days > 27:
                diz_sett["sett_0"].append(allenamento)
            elif (data_prossima_stat - data_allenamento).days > 20:
                diz_sett["sett_1"].append(allenamento)
            elif (data_prossima_stat - data_allenamento).days > 13:
                diz_sett["sett_2"].append(allenamento)
            elif (data_prossima_stat - data_allenamento).days > 6:
                diz_sett["sett_3"].append(allenamento)
            elif (data_prossima_stat - data_allenamento).days > -1:
                diz_sett["sett_4"].append(allenamento)

    # CREAZIONE X TICKS LABELS
    x_labels = [0,0,0,0]
    subtract_days_list = [27, 20, 13, 6]
    for label in range(len(x_labels)):
        data_split_1 = str(data_prossima_stat - dt.timedelta(days=subtract_days_list[label])).replace(" ", "-").split("-")
        data_split_2 = str(data_prossima_stat - dt.timedelta(days=subtract_days_list[label] - 6)).replace(" ", "-").split("-")

        if data_split_1[1] == data_split_2[1]:
            x_labels[label] = f"{data_split_1[2]} - {data_split_2[2]} {D_m_to_mesi[data_split_1[1]]}"
        else:
            x_labels[label] = f"{data_split_1[2]} {D_m_to_mesi[data_split_1[1]]} - {data_split_2[2]} {D_m_to_mesi[data_split_2[1]]}"

    # CREAZIONE Y ADDOMINALI E ALLENAMENTI
    y_addominali = [0,0,0,0,0]
    y_allenamenti = [0,0,0,0,0]

    allenamenti_counter = 0
    for k, v in diz_sett.items():

        for lista in v:

            if len(lista) == 2:
                if lista[1] == "addominali":
                    y_addominali[allenamenti_counter] += 1
                else:
                    y_allenamenti[allenamenti_counter] += 1

            else:
                y_addominali[allenamenti_counter] += 1
                y_allenamenti[allenamenti_counter] += 1

        allenamenti_counter += 1



    # aumento peso
    diz_aumenti = {"sett_1": [], "sett_2": [], "sett_3": [], "sett_4": []}

    for aumento in db["aumenti_peso"]:
        # aumento =
        """
        [
            "2021-05-01T00:00:00",
            "34k",
            "36k",
            "Crunch"
        ],
        """
        data_aumento = dt.datetime.fromisoformat(aumento[0])

        if (data_prossima_stat - data_aumento).days <= 27:

            if (data_prossima_stat - data_aumento).days > 20:
                diz_aumenti["sett_1"].append(aumento)

            elif (data_prossima_stat - data_aumento).days > 13:
                diz_aumenti["sett_2"].append(aumento)

            elif (data_prossima_stat - data_aumento).days > 6:
                diz_aumenti["sett_3"].append(aumento)

            elif (data_prossima_stat - data_aumento).days > -1:
                diz_aumenti["sett_4"].append(aumento)

        # diz_aumenti =
        """
        {
        "sett_1": [                
            "2021-05-01T00:00:00",
            "34k",
            "36k",
            "Crunch"
        ],
        ...,
        "sett_2": ...
        }
        """


    # creazione bilancio aumento peso
    y_aumenti_peso = [0, 0, 0, 0]

    allenamenti_counter = 0
    diz_somme_aumenti_peso = {"sett_1": [], "sett_2": [], "sett_3": [], "sett_4": []}

    # per sett, lista aumenti in diz_aumenti
    for k, v in diz_aumenti.items():

        for lista in v:
            # lista[1] = "34k"
            # lista[2] = "36k"
            if F_somma_numeri_in_stringa(lista[1]) < F_somma_numeri_in_stringa(lista[2]):
                diz_somme_aumenti_peso[k].append(1)

            else:
                diz_somme_aumenti_peso[k].append(0)

        y_aumenti_peso[allenamenti_counter] = sum(diz_somme_aumenti_peso[k])

        allenamenti_counter += 1

    counter_aumenti = 0
    for i in range(4):
        if i == 0:
            pass
        else:
            y_aumenti_peso[i] += y_aumenti_peso[counter_aumenti]
            counter_aumenti += 1

    y_aumenti_peso.insert(0, 0)

    # creazione messaggi telegram aumento peso

    diz_stringhe_aumenti_peso = {"sett_1": [], "sett_2": [], "sett_3": [], "sett_4": []}

    allenamenti_counter = 0
    for sett, aumenti_list in diz_aumenti.items():

        # per aumento in settimana (4 settimane)
        for aumento_list_idx in range(len(aumenti_list)):
            # creazione stringa ->
            diz_stringhe_aumenti_peso[sett].append(
                f"{diz_somme_aumenti_peso[sett][aumento_list_idx]} _{diz_aumenti[sett][aumento_list_idx][3]}_:  {diz_aumenti[sett][aumento_list_idx][1]} <b>→</b> {diz_aumenti[sett][aumento_list_idx][2]}"
            )

            if diz_stringhe_aumenti_peso[sett][aumento_list_idx][0] == "1":
                diz_stringhe_aumenti_peso[sett][aumento_list_idx] = f"💹 {diz_stringhe_aumenti_peso[sett][aumento_list_idx][2:]}"

            else:
                diz_stringhe_aumenti_peso[sett][aumento_list_idx] = f"🈹 {diz_stringhe_aumenti_peso[sett][aumento_list_idx][2:]}"


        allenamenti_counter += 1


    # GRAFICO
    x_settimane = [0, 1, 2, 3, 4]
    x_aumento_peso = [0.5,1,2,3,4]

    fig1 = plt.figure(figsize=(5,5)) #numeri: grandezza del grafico
    ax = fig1.add_axes([0.1, 0.12, 0.8, 0.8])  #stabiliamo le grandezze della barra a sinistre e sotto e del grafico stesso.  numeri: in ordine, parte sinistra, parte in basso, larghezza, altezza

    ax.grid('on')

    ax.set_title("Allenamento nelle ultime 4 settimane")
    # creazione grafico e legenda (con label)

    # x e y devono essere le x e y delle volte in cui ci si è allenati di più
    ax.axhline(y=max(y_allenamenti), color=".45", linestyle="--")

    counter_max = 1
    for numero in y_allenamenti[1:]:
        if numero == max(y_allenamenti[1:]):
            ax.axvline(x=counter_max, color=".45", linestyle="--")

        counter_max += 1



    fill_1 = ax.fill_between(x_settimane, y_allenamenti)
    fill_1.set_facecolors("#702bd150")
    fill_1.set_linewidths([3])


    ax.plot(x_aumento_peso, y_aumenti_peso, label="Aumenti peso", color="#30bb45")

    # procedura per avere rosso e verde in aumento peso
    points = np.array([x_aumento_peso, y_aumenti_peso]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cm = dict(zip([-1,0,1],["#d33434",
                            "#30bb45",
                            "#30bb45"]))
    colors = list(map(cm.get, np.sign(np.diff(y_aumenti_peso))))
    np.sign(np.diff(y_aumenti_peso))

    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    ax.plot(x_settimane, y_allenamenti, label="Allenamenti", color="#702bd1",
            lw=2, marker="o", markersize=20, markerfacecolor="#702bd175",
            markeredgecolor="#702bd1", markeredgewidth=2)
    ax.plot(x_settimane, y_addominali, label="Addominali", color="#982bd1",
            lw=2, marker="o", markersize=20, markerfacecolor="#982bd175",
            markeredgecolor="#982bd1", markeredgewidth=2)

    fill_1 = ax.fill_between(x_settimane, y_addominali)
    fill_1.set_facecolors("#982bd150")
    fill_1.set_linewidths([3])

    counter = 0
    for x, y in zip(x_settimane, y_allenamenti):
        if counter == 0:
            pass
        else:
            plt.text(x-0.05, y-0.08, str(y), color="black", fontsize=12).set_weight("bold")

        counter += 1

    counter = 0
    for x, y in zip(x_settimane, y_addominali):
        if counter == 0:
            pass
        else:
            plt.text(x-0.05, y-0.08, str(y), color="black", fontsize=12).set_weight("bold")

        counter += 1

    ax.margins(x=.1, tight=True)

    ttl = ax.title
    ttl.set_weight('bold')

    # colore bordo grafico in alto e a destra
    ax.spines['right'].set_color((1,1,1))
    ax.spines['left'].set_color((1,1,1))
    ax.spines['top'].set_color((1,1,1))

    # plt.xticks(rotation=25)
    ax.set_xticks([1, 2, 3, 4])
    # al posto di questi numeri dovrebbe esserci tipo 03 - 9 feb ecc..
    ax.set_xticklabels(x_labels)
    plt.xticks(rotation=15)


    if max(y_aumenti_peso) > max(y_allenamenti):
        maximum_y = max(y_aumenti_peso)
    else:
        maximum_y = max(y_allenamenti)


    # ax.axes.yaxis.set_ticklabels([])
    ax.set_yticklabels([])

    plt.text(3.9, max(y_aumenti_peso) - 0.1, max(y_aumenti_peso), fontsize = 12,
             bbox = dict(facecolor= '#30bb4575', boxstyle= "round",
                         edgecolor= "#30bb45", linewidth=2), fontdict=dict(weight="bold"))

    if maximum_y < 5:
        maximum_y_text = 5
    else:
        maximum_y_text = maximum_y

    ax.set_ylim(ymin=0, ymax=maximum_y_text + 1)
    ax.set_xlim(xmin=.5, xmax=4.5)


    ratio = maximum_y_text * 0.06
    plt.text(0.5, maximum_y_text - ratio*3 + 1.12, "Allenamenti", fontsize = 12, bbox= dict(facecolor= '#fff',
                                                                                                 edgecolor="#fff", pad=1.7))
    plt.text(0.5, maximum_y_text - ratio*2 + 1.12, "Addominali", fontsize = 12, bbox= dict(facecolor= '#fff',
                                                                                                edgecolor="#fff", pad=1.7))
    plt.text(0.5, maximum_y_text - ratio + 1.12, "Aumenti peso", fontsize = 12, bbox= dict(facecolor= '#fff',
                                                                                                edgecolor="#fff", pad=1.7))

    ultima_stat_str = "ultima_stat"
    y_aumenti_peso_str = "y_aumenti_peso"
    y_allenamenti_str = "y_allenamenti"
    y_addominali_str = "y_addominali"

    sum_y_allenamenti = sum(y_allenamenti[1:])
    sum_y_addominali = sum(y_addominali[1:])
    sum_y_aumenti_peso = y_aumenti_peso[-1]



    # + - in alto a destra
    if db["ultima_stat"]["y_allenamenti"] < sum_y_allenamenti:
        plt.text(1.55, maximum_y_text - ratio*3 + 1.12, f"+{sum_y_allenamenti - db[ultima_stat_str][y_allenamenti_str]}", color="#30bb45", fontsize = 12,
                 bbox= dict(facecolor= '#fff',
                 edgecolor="#fff", pad=1.7))
    elif db["ultima_stat"]["y_allenamenti"] > sum_y_allenamenti:
        plt.text(1.55, maximum_y_text - ratio*3 + 1.12, f"-{db[ultima_stat_str][y_allenamenti_str] - sum_y_allenamenti}",color= "#d33434", fontsize = 12,
                 bbox= dict(facecolor= '#fff',
                 edgecolor="#fff", pad=1.7))
    else:
        plt.text(1.55, maximum_y_text - ratio*3 + 1.12, f"invariato", fontsize = 12, color= "#3473b6",
                 bbox= dict(facecolor= '#fff',
                            edgecolor="#fff", pad=1.7))

    if db["ultima_stat"]["y_addominali"] < sum_y_addominali:
        plt.text(1.46, maximum_y_text - ratio*2 + 1.12, f"+{sum_y_addominali - db[ultima_stat_str][y_addominali_str]}", color="#30bb45", fontsize = 12,
                 bbox= dict(facecolor= '#fff',
                 edgecolor="#fff", pad=1.7))
    elif db["ultima_stat"]["y_addominali"] > sum_y_addominali:
        plt.text(1.46, maximum_y_text - ratio*2 + 1.12, f"-{db[ultima_stat_str][y_addominali_str] - sum_y_addominali}",color= "#d33434", fontsize = 12,
                 bbox= dict(facecolor= '#fff',
                 edgecolor="#fff", pad=1.7))
    else:
        plt.text(1.46, maximum_y_text - ratio*2 + 1.12, f"invariato", fontsize = 12, color= "#3473b6",
                 bbox= dict(facecolor= '#fff',
                            edgecolor="#fff", pad=1.7))

    if db["ultima_stat"]["y_allenamenti"] < sum_y_aumenti_peso:
        plt.text(1.68, maximum_y_text - ratio + 1.12, f"+{sum_y_aumenti_peso - db[ultima_stat_str][y_aumenti_peso_str]}", color="#30bb45", fontsize = 12,
                 bbox= dict(facecolor= '#fff',
                 edgecolor="#fff", pad=1.7))
    elif db["ultima_stat"]["y_allenamenti"] > sum_y_aumenti_peso:
        plt.text(1.68, maximum_y_text - ratio + 1.12, f"-{db[ultima_stat_str][y_aumenti_peso_str] - sum_y_aumenti_peso}",color= "#d33434", fontsize = 12,
                 bbox= dict(facecolor= '#fff',
                 edgecolor="#fff", pad=1.7))
    else:
        plt.text(1.68, maximum_y_text - ratio + 1.12, f"invariato", fontsize = 12, color= "#3473b6",
                 bbox= dict(facecolor= '#fff',
                            edgecolor="#fff", pad=1.7))

    path_img = f"immagini/ultimo_grafico_4sett.png"
    plt.savefig(path_img, dpi = 500)


    # send image
    img = open(path_img, 'rb')


    # MESSAGGIO GRAFICO
    msg = "<b>GRAFICO 4 SETTIMANE</b>\n"

    if db["ultima_stat"]["y_allenamenti"] < sum_y_allenamenti:
        msg += f"✦ Totale allenamenti: <code>{sum_y_allenamenti} (\+{sum_y_allenamenti - db[ultima_stat_str][y_allenamenti_str]})</code>\n"
    elif db["ultima_stat"]["y_allenamenti"] > sum_y_allenamenti:
        msg += f"✦ Totale allenamenti: <code>{sum_y_allenamenti} (-{db[ultima_stat_str][y_allenamenti_str] - sum_y_allenamenti})</code>\n"
    else:
        msg += f"✦ Totale allenamenti: <code>{sum_y_allenamenti} (=)</code>\n"
    msg += f"   Media: <code>{(str(round(sum_y_allenamenti/7, 2)))}</code>\n"

    if db["ultima_stat"]["y_addominali"] < sum_y_addominali:
        msg += f"✦ Totale addominali: <code>{sum_y_addominali} (\+{sum_y_addominali - db[ultima_stat_str][y_addominali_str]})</code>\n"
    elif db["ultima_stat"]["y_addominali"] > sum_y_addominali:
        msg += f"✦ Totale addominali: <code>{sum_y_addominali} (-{db[ultima_stat_str][y_addominali_str] - sum_y_addominali})</code>\n"
    else:
        msg += f"✦ Totale addominali: <code>{sum_y_addominali} (=)</code>\n"
    msg += f"   Media: <code>{(str(round(sum_y_addominali/7, 2)))}</code>\n"

    if db["ultima_stat"]["y_aumenti_peso"] < sum_y_aumenti_peso:
        msg += f"✦ Bilancio aumenti di peso: <code>{sum_y_aumenti_peso} (\+{sum_y_aumenti_peso - db[ultima_stat_str][y_aumenti_peso_str]})</code>\n\n"
    elif db["ultima_stat"]["y_aumenti_peso"] > sum_y_aumenti_peso:
        msg += f"✦ Bilancio aumenti di peso: <code>{sum_y_aumenti_peso} (-{db[ultima_stat_str][y_aumenti_peso_str] - sum_y_aumenti_peso})</code>\n"
    else:
        msg += f"✦ Bilancio aumenti di peso: <code>{sum_y_aumenti_peso} (=)</code>\n"
    msg += f"   Media: <code>{(str(round(sum_y_aumenti_peso/7, 2)))}</code>\n\n"


    allenamenti_counter = 0
    for k, v in diz_aumenti.items():

        msg += f"<b>Settimana: </b><code>{x_labels[allenamenti_counter]}</code>\n"
        for i in range(len(v)):
            msg += f"{diz_stringhe_aumenti_peso[k][i]}\n"

        msg += "\n"
        allenamenti_counter += 1

    return msg, img, sum_y_allenamenti, sum_y_addominali, sum_y_aumenti_peso



# GRAFICO MENSILE

def Grafico_mensile(db, F_somma_numeri_in_stringa, D_m_to_mesi, F_ora_adesso, actual_environment):

    # RAGGRUPPAMENTO IN COPPIE DI SETTIMANE
    diz_sett = {"sett2_0":  [], "sett2_1":  [], "sett2_2":  [], "sett2_3":  [], "sett2_4":  [], "sett2_5":  [], "sett2_6":  [], "sett2_7":  [], "sett2_8":  [], "sett2_9":  [],
                "sett2_10": [], "sett2_11": [], "sett2_12": [], "sett2_13": [], "sett2_14": [], "sett2_15": [], "sett2_16": [], "sett2_17": [], "sett2_18": [], "sett2_19": [],
                "sett2_20": [], "sett2_21": [], "sett2_22": [], "sett2_23": [], "sett2_24": [], "sett2_25": [], "sett2_26": []}

    # lista di numeri: -1, 14, 28, ..., 363
    lista_raggruppamento = [i for i in range(363, -2, -14)]

    data_prossima_stat_m = dt.datetime(db["data_prossima_stat_m"][0], db["data_prossima_stat_m"][1], db["data_prossima_stat_m"][2])

    i = 0
    #420 iterazioni
    for allenamento in db["allenamenti"]:
        if (data_prossima_stat_m - dt.datetime.fromisoformat(allenamento[0])).days <= 377:

            while True:
                if (data_prossima_stat_m - dt.datetime.fromisoformat(allenamento[0])).days > lista_raggruppamento[i]:
                    diz_sett[f"sett2_{i}"].append(allenamento)
                    break

                elif i == 26:
                    break

                elif (data_prossima_stat_m - dt.datetime.fromisoformat(allenamento[0])).days > lista_raggruppamento[i + 1]:
                    i += 1
                    diz_sett[f"sett2_{i}"].append(allenamento)
                    break

                else:
                    i += 1
    # print(f"{diz_sett}")


    # RAGGRUPPAMENTO AUMENTO PESO
    # dizionario come quello di prima ma creato con dict comprehension
    diz_aumenti = {f"sett2_{i}": [] for i in range(27)}

    i = 0
    for aumento in db["aumenti_peso"]:

        if (data_prossima_stat_m - dt.datetime.fromisoformat(aumento[0])).days <= 377:

            while True:
                if (data_prossima_stat_m - dt.datetime.fromisoformat(aumento[0])).days > lista_raggruppamento[i]:
                    diz_aumenti[f"sett2_{i}"].append(aumento)
                    break

                elif i == 26:
                    break

                elif (data_prossima_stat_m - dt.datetime.fromisoformat(aumento[0])).days > lista_raggruppamento[i + 1]:
                    i += 1
                    diz_aumenti[f"sett2_{i}"].append(aumento)
                    break

                else:
                    i += 1


    #CREAZIONE XLABELS
    # lista 26 0
    x_labels = [0 for _ in range(27)]

    subtract_days_list = [i for i in range(377, 12, -14)]

    # print(f"{lista_raggruppamento = }")
    for label in range(len(x_labels)):
        data_split_1 = str(data_prossima_stat_m - dt.timedelta(days=subtract_days_list[label])).replace(" ", "-").split("-")
        data_split_2 = str(data_prossima_stat_m - dt.timedelta(days=subtract_days_list[label] - 13)).replace(" ", "-").split("-")

        if data_split_1[1] == data_split_2[1]:
            x_labels[label] = f"{data_split_1[2]} - {data_split_2[2]} {D_m_to_mesi[data_split_1[1]]}"

        else:
            x_labels[label] = f"{data_split_1[2]} {D_m_to_mesi[data_split_1[1]]} - {data_split_2[2]} {D_m_to_mesi[data_split_2[1]]}"


    # CREAZIONE Y ADDOMINALI E ALLENAMENTI
    y_addominali = [0 for i in range(27)]
    y_allenamenti = [0 for i in range(27)]

    allenamenti_counter = 0
    for k, v in diz_sett.items():

        for lista in v:

            if len(lista) == 2:
                if lista[1] == "addominali":
                    y_addominali[allenamenti_counter] += 0.5
                else:
                    y_allenamenti[allenamenti_counter] += 0.5

            else:
                y_addominali[allenamenti_counter] += 0.5
                y_allenamenti[allenamenti_counter] += 0.5

        allenamenti_counter += 1


    # CREAZIONE BILANCIO AUMENTO PESO
    y_aumenti_peso = [0 for i in range(27)]
    allenamenti_counter = 0
    diz_somme_aumenti_peso = {f"sett2_{i}": 0 for i in range(27)}

    for k, v in diz_aumenti.items():

        for lista in v:
            if F_somma_numeri_in_stringa(lista[1]) < F_somma_numeri_in_stringa(lista[2]):
                diz_somme_aumenti_peso[k] += 1

            else:
                diz_somme_aumenti_peso[k] -= 1

        y_aumenti_peso[allenamenti_counter] = diz_somme_aumenti_peso[k]

        allenamenti_counter += 1



    # CREAZIONE RIASSUNTO MESI IN MESSAGGIO TELEGRAM

    # CREAZIONE DIZIONARIO MESI
    diz_mesi = {}
    for i in range(1,25):
        if i < 13:
            if i < 10:
                diz_mesi[f"0{i}_va"] = []
            else:
                diz_mesi[f"{i}_va"] = []

        else:
            i = i - 12
            if i < 10:
                diz_mesi[f"0{i}_na"] = []
            else:
                diz_mesi[f"{i}_na"] = []

    diz_aumenti_2 = copy.deepcopy(diz_mesi)


    data_prossima_stat_m = dt.datetime(db["data_prossima_stat_m"][0], db["data_prossima_stat_m"][1], db["data_prossima_stat_m"][2])

    # RAGGRUPPAMENTO ALLENAMENTI IN MESI

    anno = db["data_prossima_stat_m"][0] - 1
    anno_in_dict = "va"
    lever_anno = False
    lever_primo = True

    # inseriamo un allenamento finto all'interno di un clone di db["allenamenti"]
    # db_allenamenti_copia = copy.deepcopy(db["allenamenti"])
    db_allenamenti_copia = []
    for allenamento in db["allenamenti"]:
        if actual_environment:
            db_allenamenti_copia.append(allenamento.value)
        else:
            db_allenamenti_copia.append(allenamento)

    # print(db_allenamenti_copia)

    data_inserire = dt.datetime(db["data_prossima_stat_m"][0] - 1, db["data_prossima_stat_m"][1], 1)
    for i, allenamento in enumerate(db_allenamenti_copia):
        # print(i, allenamento)
        # print(data_inserire)

        distanza1 = (data_inserire - dt.datetime.fromisoformat(allenamento[0])).days
        distanza2 = (data_inserire - dt.datetime.fromisoformat(db_allenamenti_copia[i + 1][0])).days
        # print("AAAAAAAAAAAAAAAAA")


        if distanza1 > 0 and distanza2 < 0:
            db_allenamenti_copia.insert(i + 1, [data_inserire, "DATA_INSERITA"])
            break

    for allenamento in db_allenamenti_copia:

        data_allenamento_split = str(allenamento[0]).split("-")
        mese = data_allenamento_split[1]

        if int(mese) == db["data_prossima_stat_m"][1] or lever_anno == True:
            lever_anno = True

            if str(anno) != str(data_allenamento_split[0]):
                anno_in_dict = "na"
                anno = data_allenamento_split[0]

            if lever_primo == False:
                lever_primo = True
            else:
                diz_mesi[f"{mese}_{anno_in_dict}"].append(allenamento)

    diz_mesi_range = copy.deepcopy(diz_mesi)


    # RAGGRUPPAMENTO AUMENTO PESO

    anno = db["data_prossima_stat_m"][0] - 1
    anno_in_dict = "va"
    lever_anno = False

    # inseriamo un allenamento finto all'interno di un clone di db["allenamenti"]
    db_aumenti_copia = []
    for aumento in db["aumenti_peso"]:
        if actual_environment:
            db_aumenti_copia.append(aumento.value)
        else:
            db_aumenti_copia.append(aumento)

    for i, aumento in enumerate(db_aumenti_copia):
        distanza1 = (data_inserire - dt.datetime.fromisoformat(aumento[0])).days
        distanza2 = (data_inserire - dt.datetime.fromisoformat(db_aumenti_copia[i + 1][0])).days

        if distanza1 > 0 and distanza2 < 0:
            db_aumenti_copia.insert(i + 1, [data_inserire, "DATA_INSERITA"])
            break


    for aumento in db_aumenti_copia:

        data_allenamento_split = str(aumento[0]).split("-")
        mese = data_allenamento_split[1]

        if int(mese) == db["data_prossima_stat_m"][1] or lever_anno == True:
            lever_anno = True

            if str(anno) != str(data_allenamento_split[0]):
                anno_in_dict = "na"
                anno = data_allenamento_split[0]

            diz_aumenti_2[f"{mese}_{anno_in_dict}"].append(aumento)

    # eliminazione a diz mesi range che mantiene 12 mesi
    list_diz = [diz_aumenti_2, diz_mesi_range]
    for diz in list_diz:

        del_list = []
        # lever_diz_mesi_r = False
        # lever_diz_mesi_r_fine = False
        lever_range = False

        for k, v in diz.items():
            # print(k,v)
            if int(k[0:2]) == db["data_prossima_stat_m"][1]:
                if lever_range == False:
                    lever_range = True
                else:
                    lever_range = False

            elif lever_range:
                pass

            else:
                del_list.append(k)

        for k in del_list:
            del diz[k]



    # print(diz_mesi)
    # for k in diz_mesi:
    #     print(k, len(diz_mesi[k]))


    # CREAZIONE Y ADDOMINALI E ALLENAMENTI
    y_addominali_2 = [0 for i in range(13)]
    y_allenamenti_2 = [0 for i in range(13)]

    allenamenti_counter = 0
    diz_mesi_range_13 = copy.deepcopy(diz_mesi_range)
    # diz_mesi_range_13 = diz_mesi_range_13[]


    for k, v in diz_mesi_range.items():
        # print(k, v)
        # print(allenamenti_counter)

        if v == []:
            # print(v, "IF")
            y_allenamenti_2[allenamenti_counter] = 0
            y_addominali_2[allenamenti_counter] = 0

        else:
            # print(v, "ELSE")
            for lista in v:

                if lista[1] != "DATA_INSERITA":
                    if len(lista) == 2:
                        if lista[1] == "addominali":
                            y_addominali_2[allenamenti_counter] += 1
                        else:
                            y_allenamenti_2[allenamenti_counter] += 1

                    else:
                        y_addominali_2[allenamenti_counter] += 1
                        y_allenamenti_2[allenamenti_counter] += 1


        allenamenti_counter += 1


    # CREAZIONE BILANCIO AUMENTO PESO
    y_aumenti_peso_2 = [0 for i in range(13)]
    allenamenti_counter = 0

    diz_somme_aumenti_peso_2 = copy.deepcopy(diz_mesi_range)
    for k in diz_somme_aumenti_peso_2:
        diz_somme_aumenti_peso_2[k] = 0


    for k, v in diz_aumenti_2.items():

        if k in diz_somme_aumenti_peso_2:

            for lista in v:
                if lista[1] == "DATA_INSERITA":
                    pass
                elif F_somma_numeri_in_stringa(lista[1]) < F_somma_numeri_in_stringa(lista[2]):
                    diz_somme_aumenti_peso_2[k] += 1
                elif F_somma_numeri_in_stringa(lista[1]) > F_somma_numeri_in_stringa(lista[2]):
                    diz_somme_aumenti_peso_2[k] -= 1


            y_aumenti_peso_2[allenamenti_counter] = diz_somme_aumenti_peso_2[k]

            allenamenti_counter += 1



    diz_stringhe_aumenti_peso_2 = {}

    allenamenti_counter = 0
    for k, v in diz_mesi_range.items():

        mese = int(k[:2])
        if k[3:] == "va":
            anno = db["data_prossima_stat_m"][0] - 1
        else:
            anno = db["data_prossima_stat_m"][0]

        tot_giorni_mese = calendar.monthrange(anno, mese)[1]

        diz_stringhe_aumenti_peso_2[k] = [y_allenamenti_2[allenamenti_counter], str(round(y_allenamenti_2[allenamenti_counter]/tot_giorni_mese*7, 2)),
                                          y_addominali_2[allenamenti_counter], str(round(y_addominali_2[allenamenti_counter]/tot_giorni_mese*7, 2)),
                                          y_aumenti_peso_2[allenamenti_counter], str(round(y_aumenti_peso_2[allenamenti_counter]/tot_giorni_mese*7, 2))]


        allenamenti_counter += 1


    n_to_mesi = {"01": "🧊Gennaio", "02": "🌲Febbraio", "03": "🐤Marzo", "04": "🇭🇰Aprile", "05": "🌸Maggio", "06": "☀️Giugno", "07": "🦅Luglio", "08": "🌴Agosto", "09": "📚Settembre", "10": "🎃Ottobre", "11": "🍂Novembre", "12": "☃️Dicembre"}

    msg = "<b>GRAFICO MENSILE</b>\n"

    for k, v in diz_stringhe_aumenti_peso_2.items():

        if k[3:] == "va":
            anno = db["data_prossima_stat_m"][0] - 1
        else:
            anno = db["data_prossima_stat_m"][0]

        msg += f"<b>{n_to_mesi[k[:2]]} {anno}</b>\n" \
                       f"Tot allenamenti: <code>{diz_stringhe_aumenti_peso_2[k][0]}</code>, media settimanale: <code>{diz_stringhe_aumenti_peso_2[k][1]}</code>\n" \
                       f"Tot addominali: <code>{diz_stringhe_aumenti_peso_2[k][2]}</code>, media settimanale: <code>{diz_stringhe_aumenti_peso_2[k][3]}</code>\n" \
                       f"Tot aumenti peso: <code>{diz_stringhe_aumenti_peso_2[k][4]}</code>, media settimanale: <code>{diz_stringhe_aumenti_peso_2[k][5]}</code>\n\n"




    # GRAFICO
    # prendiamo 26 coppie di settimane in considerazione (27 per la settimana prima a x 0)
    x_2settimane = [i for i in range(27)]

    # media di allenamenti in coppia di settimane, il primo numero è l'ultimo della settimana prima

    # mettiamo la media del bilancio di aumento peso delle due settimane, stessa cosa di prima per il primo numero


    fig1 = plt.figure(figsize=(17,5)) #numeri: grandezza del grafico
    ax = fig1.add_axes([0.04, 0.15, 0.95, 0.8])  #stabiliamo le grandezze della barra a sinistre e sotto e del grafico stesso.  numeri: in ordine, parte sinistra, parte in basso, larghezza, altezza

    ax.grid('on')

    ax.set_title("Allenamento nell'ultimo anno, diviso in coppie di settimane")
    # creazione grafico e legenda (con label)

    if max(y_aumenti_peso) > max(y_allenamenti):
        maximum_y = max(y_aumenti_peso)
    else:
        maximum_y = max(y_allenamenti)
    for i in range(math.floor(maximum_y) + 1):
        ax.axhline(y=i, color=".70", linestyle="-", lw=1)

    #INSERIMENTO DI CAMBIO SCHEDA ALL'INTERNO DEL GRAFICO TRAMITE LE LABEL
    for cambio in db["data_cambioscheda"]:
        if (F_ora_adesso() - dt.datetime.fromisoformat(cambio)).days < 370:

            data_cambioscheda_split = str(cambio).split("-")
            # print(f"{data_cambioscheda_split = }")
            mese = D_m_to_mesi[data_cambioscheda_split[1]]

            mese_prima_raw = str(int(data_cambioscheda_split[1]) - 1)
            mese_dopo_raw = str(int(data_cambioscheda_split[1]) + 1)

            if mese == "Dic":
                mese_prima = "Nov"
                mese_dopo = "Gen"
            elif mese == "Gen":
                mese_prima = "Dic"
                mese_dopo = "Feb"
            else:
                if len(mese_prima_raw) == 1:
                    mese_prima = D_m_to_mesi[f"0{mese_prima_raw}"]
                else:
                    mese_prima = D_m_to_mesi[mese_prima_raw]

                if len(mese_dopo_raw) == 1:
                    mese_dopo = D_m_to_mesi[f"0{mese_dopo_raw}"]
                else:
                    mese_dopo = D_m_to_mesi[mese_dopo_raw]


            counter_label = 0
            for i, label in enumerate(x_labels):
                if i == 0:
                    pass

                else:
                    if mese in label:

                        giorni_list = [0,0]

                        if counter_label == 0:
                            giorni_list[0] = 1

                            if mese_prima in label:
                                giorni_list[1] = int(label[9:11])
                            else:
                                giorni_list[1] = int(label[5:7])

                        else:
                            giorni_list[0] = int(label[0:2])

                            if mese_dopo in label:

                                list_mesi_anno = list(diz_mesi_range.keys())
                                del list_mesi_anno[0]

                                diz_mesi_anno = {list_mesi_anno[i][0:2]: list_mesi_anno[i][3:] for i in range(len(list_mesi_anno))}
                                # print(f"{diz_mesi_anno = }")
                                # print(f"{data_cambioscheda_split[1] = }")

                                if diz_mesi_anno[data_cambioscheda_split[1]] == "na":
                                    anno = db["data_prossima_stat_m"][0]
                                else:
                                    anno = db["data_prossima_stat_m"][0] - 1

                                giorni_list[1] = calendar.monthrange(anno, int(data_cambioscheda_split[1]))[1]

                            else:
                                giorni_list[1] = int(label[5:7])

                        for i_2 in range(giorni_list[0], giorni_list[1] + 1):
                            if i_2 == int(data_cambioscheda_split[2][0:2]):
                                ax.axvline(x=i, color="#d12bc0", lw= 4)


                        counter_label += 1



    # ax.axvline(x=15, color="#d12bc0", lw= 4)

    fill_1 = ax.fill_between(x_2settimane, y_allenamenti)
    fill_1.set_facecolors("#702bd150")
    fill_1.set_linewidths([3])

    ax.plot(x_2settimane, y_aumenti_peso, label="Bilancio kg", color="#30bb45")

    # procedura per avere rosso e verde in aumento peso
    points = np.array([x_2settimane, y_aumenti_peso]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cm = dict(zip([-1,0,1],["#d33434",
                            "#30bb45",
                            "#30bb45"]))
    colors = list(map(cm.get, np.sign(np.diff(y_aumenti_peso))))
    np.sign(np.diff(y_aumenti_peso))

    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    # print(f"{x_2settimane = }")
    # print(f"{y_allenamenti = }")
    # print(f"{y_addominali = }")
    ax.plot(x_2settimane, y_allenamenti, label="Allenamenti", color="#702bd1", lw=2)
    ax.plot(x_2settimane, y_addominali, label="Addominali", color="#982bd1", lw=2)


    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    fill_1 = ax.fill_between(x_2settimane, y_addominali)
    fill_1.set_facecolors("#982bd150")
    fill_1.set_linewidths([3])

    ax.margins(x=.1, tight=True)

    ttl = ax.title
    ttl.set_weight('bold')

    # colore bordo grafico in alto e a destra
    ax.spines['right'].set_color((.7,.7,.7))
    ax.spines['top'].set_color((.7,.7,.7))

    plt.xticks(rotation=25)
    ax.set_xticks([i for i in range(27)])
    ax.set_xticklabels(x_labels)

    ax.set_ylim(ymin=0, ymax=maximum_y + 1)
    ax.set_xlim(xmin=.5, xmax=26.5)


    path_img = f"immagini/ultimo_grafico_mensile.png"
    plt.savefig(path_img, dpi=400)


    # SEND
    img = open(path_img, 'rb')

    return msg, img


# SCHEDA TO IMAGE

def schedaIndex_to_image(diz_all, scheda_YYYY_MM_DD):
    active_workouts = [key for key, val in diz_all.items() if val != []]

    workbook = xlsxwriter.Workbook('immagini/creazione_img-scheda.xlsx')
    worksheet = workbook.add_worksheet()

    # FORMATS

    rescale_factor = 0.8  # questo numero viene applciato ad ogni font size in modo che il testo sia più piccolo ma che mantenga la proporzione

    # bianco
    normal_format = workbook.add_format()

    right_align_format = workbook.add_format()
    right_align_format.set_align("right")

    bold_left_align_format = workbook.add_format()
    bold_left_align_format.set_align("left")

    italic_format = workbook.add_format({"italic": True})

    indent_format = workbook.add_format()
    indent_format.set_indent(2)

    # grigio
    grey_normal_format = workbook.add_format()

    grey_italic_format = workbook.add_format({"italic": True})

    grey_right_align_format = workbook.add_format()
    grey_right_align_format.set_align("right")

    grey_bold_left_align_format = workbook.add_format()
    grey_bold_left_align_format.set_align("left")

    grey_indent_format = workbook.add_format()
    grey_indent_format.set_indent(2)

    # titoli
    title_format = workbook.add_format({"bold": True})
    title_format.set_font_size(22)

    workout_title_format = workbook.add_format({"bold": True})
    workout_title_format.set_font_size(18)
    workout_title_format.set_indent(2)


    # facciamo in modo che tutti i format abbiano sfondo bianco o grigio
    white_formats = [normal_format, title_format, workout_title_format, indent_format, right_align_format, bold_left_align_format, italic_format]
    [format_.set_bg_color("FFFFFF") for format_ in white_formats]
    w_formats_dict = {"normal": normal_format, "indent": indent_format, "italic": italic_format, "right": right_align_format, "b_left": bold_left_align_format}

    grey_formats = [grey_italic_format, grey_normal_format, grey_indent_format, grey_right_align_format, grey_bold_left_align_format]
    [format_.set_bg_color("EFEFEF") for format_ in grey_formats]
    g_formats_dict = {"normal": grey_normal_format, "indent": grey_indent_format, "italic": grey_italic_format, "right": grey_right_align_format, "b_left": grey_bold_left_align_format}

    all_formats = set(white_formats).union(set(grey_formats))
    for format in all_formats:
        format.set_font_size(format.font_size*rescale_factor)


    def write_blank_line(last_row):
        [worksheet.write(last_row, i, None, normal_format) for i in range(6)]

    col = 0
    last_row = 0  # questa variabile rappresenta l'ultima riga a cui siamo arrivati, su excel sarebbe last_row+1
    fontsize_exceptions = dict()  # questo dizionario rappresenta quali righe (keys) non hanno una font size normale

    """ SCHEDA 2022... """
    anno_inizio, sett, data_inizio, data_fine, i, ultima_scheda = schedaFname_to_title(scheda_YYYY_MM_DD)
    if ultima_scheda == False:
        titolo = f"SCHEDA {anno_inizio} {weekday_nday_month_year(data_inizio)} - {weekday_nday_month_year(data_fine)}, {sett} sett"
    else:
        titolo = f"SCHEDA ATTUALE: {weekday_nday_month_year(data_inizio)}, {sett} sett"

    worksheet.merge_range("A1:F1", titolo, title_format)
    fontsize_exceptions[last_row] = title_format.font_size
    last_row += 1

    write_blank_line(last_row)
    last_row += 1

    for workout in active_workouts:
        worksheet.merge_range(f"A{last_row+1}:F{last_row+1}", nomi_in_scheda[workout].upper(), workout_title_format)  # aggiungiamo 1 perchè excel è 1-indexed e xls writer 0-indexed
        fontsize_exceptions[last_row] = workout_title_format.font_size
        last_row += 1

        for i, ese in enumerate(diz_all[workout]):  # iterazione per righe, ad ogni esercizio
            flick = i%2
            if flick == 0:
                bg_format = w_formats_dict
            else:
                bg_format = g_formats_dict

            worksheet.write(last_row+i, col+0, f"{diz_all[workout][i][0]}\"", bg_format["indent"])  # pausa
            worksheet.write(last_row+i, col+1, diz_all[workout][i][1], bg_format["italic"])  # nome
            worksheet.write(last_row+i, col+2, diz_all[workout][i][2], bg_format["right"])  # rep
            worksheet.write(last_row+i, col+3, f' {diz_all[workout][i][3]}', bg_format["b_left"])  # serie
            worksheet.write(last_row+i, col+4, diz_all[workout][i][4], bg_format["normal"])  # peso
            worksheet.write(last_row+i, col+5, diz_all[workout][i][5], bg_format["italic"])  # note

        last_row += i+1  # aggiungiamo 1 perchè i parte da 0
        write_blank_line(last_row)
        last_row += 1

    colonna_tempo_fs_11 = 6
    colonna_serie_fs_11 = 3
    # queste due variabili indicano quanto deve essere lunga la cella del tempo e delle serie quando la font size è 11. usando il rescaling factor possiamo adattarle ad un altra fs

    worksheet.set_column(0, 0, colonna_tempo_fs_11*rescale_factor)
    worksheet.set_column(3, 3, colonna_serie_fs_11*rescale_factor)

    # AUTO FITTING
    # questo codice è per la parte di aggiustamento dell'altezza delle celle e della loro lunghezza
    # questo for loop serve per fare in modo che l'altezza delle celle corrisponda con la font size
    all_cell_heights_sum = 0  # questo numero è la somma di tutte le lunghezze delle celle e ci servirà per tagliare (in verticale) l'immagine finale più avanti nel codice

    for i in range(last_row):
        fontSize_cellHeight_ratio = 1.275  # significato di questo numero: su excel se la font size è 10, la cell height sarà round(10*1,275)
        font_size = 9

        if i in fontsize_exceptions:  # se una delle keys è i
            font_size = fontsize_exceptions[i]

        cell_height = math.ceil(font_size*fontSize_cellHeight_ratio)
        worksheet.set_row(i, cell_height)
        all_cell_heights_sum += cell_height


    # troviamo quanti caratteri ha la colonna che dovrebbe essere la più lunga
    longest_lenght = {1: 1, 2: 1, 4: 1, 5: 1}  # le chiavi sono le colonne a cui si metterà l'autofitting
    for column in [1, 2, 4, 5]:  # le colonne a cui dobbiamo modificare la lungezza usando autofitting. sono le colonne di: nome, reps, peso, note
        for workout in active_workouts:
            for ese in diz_all[workout]:
                longest_lenght[column] = max(longest_lenght[column], len(str(ese[column]).replace("\\", "")))

    lenght_rescaling_factor = {1: 0.78,
                               2: 0.9,
                               4: 1,
                               5: 0.78}  # questi numeri servono per portare da lunghezza in caratteri a valore di lunghezza della cella. sono applicati diversamente ad ogni colonna,
    # visto che ad esempio i numeri occupano più spazio del nome degli esercizi (es: "000" non ha la stessa lunghezza di "Alz" su excel).
    for column_k, column_len in longest_lenght.items():
        worksheet.set_column(column_k, column_k, math.ceil(column_len*lenght_rescaling_factor[column_k]*rescale_factor))

    workbook.close()


    # DA XLSX A IMMAGINE FINALE

    # XLSX TO PDF
    convertapi.api_secret = 'W37nRGb3Id0teFuD'
    convertapi.convert('pdf', {'File': 'immagini/creazione_img-scheda.xlsx'}, from_format='xlsx').save_files('immagini/creazione_img-scheda.pdf')


    pdf = pdfium.PdfDocument("immagini/creazione_img-scheda.pdf")
    n_pages = len(pdf)
    np_pdf_pages = []

    def index_range_grouping(lista):
        """
        a = [-1, 0, 2, 3, 4, 8, 9, 10, 11, 13]  ->  b = [[-1, 0], [2, 4], [8, 11], [13, 13]]
        """
        list_ranges = []

        for i, num in enumerate(lista):
            if i == 0:
                start = num
            elif num == lista[i-1] + 1:
                pass
            else:
                end = lista[i-1]
                list_ranges.append([start, end])
                start = num

        list_ranges.append([start, num])  # ultimo elemento

        return list_ranges


    def range_to_numbers(range_list):
        # inverso di index_range_grouping
        numbers_list = []

        for range_ in range_list:
            [numbers_list.append(num) for num in range(range_[0], range_[1]+1)]

        return numbers_list


    # PDF TO IMG
    img_scale = 3  # questo numero determina la lenght e width dell'immagine derivante dal pdf

    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        pil_image = page.render_topil(scale=img_scale,
                                      rotation=0,
                                      crop=(0, 0, 0, 0),
                                      greyscale=True,
                                      optimise_mode=pdfium.OptimiseMode.NONE)
        # pil_image.save(f"image_{page_number+1}.png")
        np_pdf_pages.append(np.array(pil_image))

    # RACCOGLIMENTO INDEX DELLE COLONNE (RIGHE VERTICALI) COMPLETAMENTE BIANCHE
    pdf_shape = np_pdf_pages[0].shape
    height = pdf_shape[0]
    width = pdf_shape[1]
    white_vertical_row = np.full(height, 255)
    blank_columns_indexes = [[] for _ in np_pdf_pages]
    blank_columns_ranges = copy.deepcopy(blank_columns_indexes)

    for i_page, page in enumerate(np_pdf_pages):
        for column in range(width):
            if (page[:, column] == white_vertical_row).all():
                blank_columns_indexes[i_page].append(column)

    for i_page in range(len(blank_columns_ranges)):  # raggruppamento degli index
        blank_columns_ranges[i_page] = index_range_grouping(blank_columns_indexes[i_page])
    """ [[[0, 101], [645, 1190]], [[0, 101], [1093, 1190]], [[0, 101], [106, 111], [171, 176], [235, 235], [251, 1190]]] """  # come si può vedere alcuni range nella terza pagina
    # sono molto piccoli, questi range di 5 pixel sono lo spazio, es: "crunch cavi" lo spazio tra crunch e cavi è un range di pixel bianchi

    # RIMOZIONE RANGE PICCOLI
    minimum_range_width = 20    # solo i range maggiori di 20 verranno tenuti
    suitable_deletion_ranges = [[] for _ in blank_columns_indexes]

    for i_page, page in enumerate(blank_columns_ranges):
        for i_range, range_ in enumerate(page):
            if range_[1] - range_[0] > minimum_range_width:
                suitable_deletion_ranges[i_page].append(range_)

        # teniamo il bordo della prima pagina
        if i_page == 0:
            del suitable_deletion_ranges[0][0]

        # teniamo solo gli index che soddisfano i requisiti
        blank_columns_ranges[i_page] = [range_ for range_ in suitable_deletion_ranges[i_page]]

    for i_page in range(len(np_pdf_pages)):
        deletion_indexes = range_to_numbers(blank_columns_ranges[i_page])

        # np_pdf_pages[i_page] = np.delete(np_pdf_pages[i_page], blank_columns_ranges_tuple, 1)  # inserendo una tuple nel parametro del primo index rimuoviamo tutti quegli index insieme
        np_pdf_pages[i_page] = np.delete(np_pdf_pages[i_page], deletion_indexes, 1)  # inserendo una tuple nel parametro del primo index rimuoviamo tutti quegli index insieme


    concatenated_img = np.concatenate(tuple(np_pdf_pages), axis=-1)  # concateniamo i vari numpy array per formare una immagine unica
    img = Image.fromarray(concatenated_img)

    # CROPPING
    top_and_left_pixels_scale_1 = 42  # questo numero rappresenta il pixel da cui partiamo a tagliare sia a sinistra dell'immagine che in alto usando img_scale = 1
    left_top_rescaling_factor = 181 / 161  # questo numero ci permette di sapere da che pixel partire dall'alto sapendo il pixel da ci siamo partiti da sinistra,
    #   questo perchè il padding dei pdf a sinistra è più piccolo di quello dell'alto ma mantengono comunque una proporzione in base alla img_scale
    left_first_pixel = top_and_left_pixels_scale_1*img_scale
    top_first_pixel = left_first_pixel*left_top_rescaling_factor
    height_cropping_factor = 1.25  # questo numero rappresenta di quanto moltiplicare all_cell_heights_sum*rescale_factor*img_scale per ottenere un cropping che includa tutti gli
    #   esercizi
    original_width, _ = img.size
    img = img.crop((left_first_pixel, top_first_pixel, original_width,
                   all_cell_heights_sum*rescale_factor*img_scale*height_cropping_factor + left_first_pixel))  # + left_first_pixel ha l'obiettivo di dare lo stesso padding che
    #   c'è a destra (e anche in alto)

    try:
        os.remove("immagini/creazione_img-scheda.xlsx")
    except:
        pass

    img_path = f"schede/immagini/{scheda_YYYY_MM_DD}.jpeg"
    img.save(img_path)
    loaded_img = open(img_path, 'rb')
    return loaded_img


# GRAFICO WORKOUT SCHEDA

def Grafico_radarScheda(dizAll_inizio, dizAll_fine, date_workout, count_workout):
    ic(date_workout)
    def angle2rads(angle):
        return angle/(180/math.pi)

    def radian_to_rotation(radian):
        angle = radian*180/math.pi

        # lato in cui è l'angolo
        if angle > 0 and angle < 180:
            rotation = "left"
        else:
            rotation = "right"

        # se l'angolo è da 207 a 153 la rotaizone è center
        if angle < 181 and angle > 179:
            rotation = "center"
        elif angle < 1 or angle > 365:  # anche se è tra 27 e 333
            rotation = "center"

        return rotation

    def angle_to_xy(radians, max_y):
        """ questa fuzione prendendo i radians e il valore massimo di y ci dice le x ed y del testo che vogliamo aggiungere """
        added_x = [-5.5, 0, 5.5]  # significato: a 90 o 270 gradi vengono aggiunti -2.9 a x0, 0 a x1, e 2.9 a x2, dove x è il testo
        added_y = [11, 7, 3]  # stesso significato ma a 0 e 180 gradi e con y
        rescaling_factor = max_y/40  # dividiamo per 40 perchè le dimensioni iniziali sono state create con un max_y di 40, quindi ad esempio se dovesse esserci 80 dobbiamo
            # moltiplicare per 2
        added_y = [y*rescaling_factor for y in added_y]  # scaliamo la y in base a quanto è la y massima

        angle = radians*(180/math.pi)

        # X Y FACTORS

        direction = "left"
        x_inversion = True
        if angle >= 0 and angle < 180:
            direction = "right"
            x_inversion = False

        y_inversion = False
        if angle > 90 and angle < 270:  # se siamo sotto alla metà del cerchio dobbiamo applicare l'inversione di y0, y1, y2 (quindi y2, y1, y0)
            y_inversion = True

        sideCenter_proximity = angle
        if direction == "left":
            sideCenter_proximity -= 180
        if sideCenter_proximity > 90:
            sideCenter_proximity = abs(sideCenter_proximity - 180)  # es: angle = 170, sideCenter_proximity = 10, questo vuol dire che su una scala da 0 a 90 siamo vicini 10 a 90
            # ic(sideCenter_proximity)

        x_factor = sideCenter_proximity / 90  # parte da 0 a x = 0 e arriva a 1 ad x = 90 o 270, poi
        y_factor = 1 - x_factor  # opposto di x_factor, più gradi di vicinanza a sidemax ci sono più è basso, perchè se siamo vicini a sidemax dobbiamo aggiugnere solo x e non y

        # OBLIQUE FACTOR
        # questo fattore ci serve perchè il testo che è nei punti obliqui (45°, 135°, ...) è più attaccato di quello che è nei punti dritti (nord ovest est..)

        # troviamo qual è il punto obliquo più vicino
        obliques = [45, 135, 225, 315]
        sub_obliques = [abs(angle - ob) for ob in obliques]
        nearest_idx = sub_obliques.index(min(sub_obliques))  # otteniamo il numero più piccolo che di conseguenza è il lato obliquo più vicino

        oblique_proximity = abs(angle - 90*obliques.index(obliques[nearest_idx]))
        if obliques[nearest_idx] < angle:
            oblique_proximity = abs(90*(obliques.index(obliques[nearest_idx]) + 1) - angle)

        oblique_factor = oblique_proximity/45

        # AGGIUNTA COORDINATE

        added_x_oblique = [a_x*(oblique_factor*0.16) for a_x in added_x]
        added_x = [a_x*x_factor + a_x_o for a_x, a_x_o in zip(added_x, added_x_oblique)]
        added_y_oblique = [a_y*(oblique_factor*0.16) for a_y in added_y]
        added_y = [a_y*y_factor + a_y_o for a_y, a_y_o in zip(added_y, added_y_oblique)]
        if y_inversion:
            added_y.reverse()
        if x_inversion:
            added_x.reverse()


        XYs = [[radians+angle2rads(a_x), max_y+a_y+(2.5*x_factor)] for a_x, a_y in zip(added_x, added_y)]  # ha anche una costante ( 2.5*x_factor) perchè in questo modo è
            # simmetrico
        return XYs

    # DATI INIZIALI
    data_oggi = ora_EU()

    radarGraph_diz = {workout: {} for workout, esercizi in dizAll_fine.items() if esercizi != [] and workout != "addominali" and count_workout[workout] > 1}  # if esercizi !=
    ic(radarGraph_diz)
        # []: se era un active workout. non ci serve il grafico degli addominali e non possiamo usare gli esercizi che non sono stati fatti almeno 2 volte
    active_workouts = [workout for workout in radarGraph_diz]
    num_workouts = len(active_workouts)

    for workout in active_workouts:
        for i_ese, ese in enumerate(dizAll_fine[workout]):
            # ic(ese)
            radarGraph_diz[workout][ese[1]] = [stringaPeso_to_num(dizAll_inizio[workout][i_ese][4]), stringaPeso_to_num(ese[4])]
    """ radarGraph_diz = {workout1: {nome_esercizio1: [peso_dopo_seconda_settimana, peso_al_cambio_scheda], ...}, workout2: ...} """


    # PLOT

    fig = plt.figure(figsize=(7.5, 5*num_workouts))

    for workout_i, workout in enumerate(active_workouts):

        sns.set_style("darkgrid", {"grid.linestyle": "-", "grid.color": "0.75"})
        ax = fig.add_subplot(num_workouts, 1, workout_i+1, polar=True)  # polar=True fa in modo che il plot diventa un cerchio e che x kg_i trasformi in radians, un angolo di 360° è uguale a
        # pi greco * 2

        # DATI
        nomi_esercizi = [ese for ese in radarGraph_diz[workout]]
        kg_inizio = [peso[0] for _, peso in radarGraph_diz[workout].items()]
        kg_fine = [peso[1] for _, peso in radarGraph_diz[workout].items()]

        num_esercizi = len(nomi_esercizi)
        max_kg = max(max(kg_inizio), max(kg_fine))
        x_radians = np.linspace(0, 2*np.pi, num_esercizi, endpoint=False)  # angoli espressi in radians (angolo di 360° in radians = 2 * pi greco)

        # concatenazione per ottenere il plot completo con ax.fill
        kg_inizio = np.concatenate((kg_inizio, [kg_inizio[0]]))
        kg_fine = np.concatenate((kg_fine, [kg_fine[0]]))
        x_radians = np.concatenate((x_radians, [x_radians[0]]))
        nomi_esercizi.append(nomi_esercizi[0])



        ic(date_workout[workout])
        ic(data_oggi)
        # FILL
        ax.fill(x_radians, kg_inizio, alpha=0.35, zorder=1, color="#ec5e92", linewidth=0, label=weekday_nday_month_year(date_workout[workout], year=True))
        ax.fill(x_radians, kg_fine, alpha=0.35, zorder=1, color="#8e68dc", linewidth=0, label=weekday_nday_month_year(data_oggi, year=True))

        # LEGENDA
                                            #    X,   Y    espressi in percentuale
        ax.legend(frameon=False, bbox_to_anchor=(1.45, 1.25, 0, 0), alignment="right")


        # BANDE CIRCOLARI Y (KG) E Y LABELS

        kg_labels = ax.get_yticklabels()  # otteniamo le label che matpltolib ha creato per il peso
        Ys = [kg_labels[i].get_position()[1] for i in range(len(kg_labels))]

        ax.grid(False)

        x_circle_values = np.linspace(0, 2*math.pi, 100, endpoint=True)
        for i in range(1, len(Ys), 2):
            ax.fill_between(x_circle_values,
                            y1=[Ys[i-1] for _ in range(len(x_circle_values))],
                            y2=[Ys[i] for _ in range(len(x_circle_values))],
                            color="0", alpha=0.07, zorder=0, linewidth=0)

        # labels e style linee KG
        [ax.axvline(x, c="0.7", linewidth=1.5, zorder=1) for x in x_radians]
        ax.set_rgrids([y for y in Ys], [f"{kg_label.get_text()}kg" for kg_label in kg_labels], fontfamily="monospace", fontsize=7)


        # LINEE ROSSE/VERDI E LISTE TESTO DIMINUZIONI / AUMENTI

        colors = ["red" for _ in range(num_esercizi)]  # partiamo dalla condizinoe che c'è stata una perdita
        cambiamento = [True for _ in range(num_esercizi)]
        alphas = [0 for _ in range(num_esercizi)]

        labels = [[] for _ in nomi_esercizi]
        text_colors = []
        for i, (kg_i, kg_f) in enumerate(zip(kg_inizio[:-1], kg_fine[:-1])):

            # LINEE Y
            diff = kg_f - kg_i

            # partiamo dalla condizione in cui non c'è stato nè aumento nè diminuzione
            text_color = "slategray"
            cambiamento[i] = False

            if diff > 0:
                colors[i] = "limegreen"
                text_color = "green"
                cambiamento[i] = True

            elif diff < 0:
                text_color = "firebrick"
                cambiamento[i] = True

            ratio = diff/kg_i if diff >= 0 else diff/kg_i

            alpha = (abs(ratio))*2  # vuol dire che se alla fine abbiamo un peso 50% superiore abbiamo alpha = 1
            alpha = max(min(1, alpha), 0.4)  # non può essere più grande di 1 e minore di 0.4
            alphas[i] = alpha

            # CREAZIONE LABELS
            labels[i].append(md_v2_replace(nomi_esercizi[i], reverse=True))  # "Panca piana"
            labels[i].append(f"{kg_i}kg → {kg_f}kg")                    # "20kg -> 25kg"
            labels[i].append(f"{diff:+.2f}kg{f', {ratio:+.2%}' if cambiamento[i] else ''}")
            text_colors.append(text_color)

        [ax.plot([x_radians[i], x_radians[i]], [kg_inizio[i], kg_fine[i]], c=colors[i], alpha=alphas[i], zorder=3) for i in range(num_esercizi) if cambiamento[i] == True]

        # AGGIUSTAMENTI

        # facciamo in modo che la prima label sia in alto e che ci sia il senso orario (altrimenti ci sarebbe senso antiorario e la prima label sarebeb a destra)
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        x_angles = x_radians * 180/math.pi  # trasformazione da radians a gradi. per convertire da radians a gradi ci basta fare * 57.29~, oppure (più semplicemente) 180/pi_greco
        ax.set_thetagrids(x_angles, ["" for _ in nomi_esercizi])  # rimozione di xticklabels

        # set limiti Y
        border_y = max_kg + max_kg/15
        ax.set_rmax(border_y)
        ax.set_rmin(0)


        # TESTO A 3 RIGHE DI OGNI ESERCIZIO
        for idx, x_radian in enumerate(x_radians[:-1]):
            XYs = angle_to_xy(x_radian, border_y)
            ha = radian_to_rotation(XYs[1][0])  # XYs[1][0] usiamo solo il punto centrale per ottenere la rotazione, altrimenti potremmo avere una parte rotata in un certo modo
                # e l'altra in un altro modo

            ax.text(XYs[0][0], XYs[0][1], f"{labels[idx][0]}",
                    ha=ha, va="center", fontsize="large", fontstyle="italic")
            ax.text(XYs[1][0], XYs[1][1], f"{labels[idx][1]}",
                    ha=ha, va="center")
            ax.text(XYs[2][0], XYs[2][1], f"{labels[idx][2]}", color=text_colors[idx],
                    ha=ha, va="center")

        # TITOLO A SX con sottotitolo
        ax.text(angle2rads(306.5), max_kg*2.4, nomi_in_scheda[workout], fontsize=18, fontweight=600, ha="left")
        days_in_between = (dt.datetime.fromisoformat(date_workout[workout]) - data_oggi).days
        ic(days_in_between)
        # mesi, sett = days_to_mesiSett(days_in_between)
        sett, giorni = (days_in_between // 7, days_in_between % 7)
        sottotitolo = f"{count_workout[workout]} volte in " + (f"{sett} sett" if sett > 0 else "") + (f" e {giorni} giorni" if giorni > 0 else "")
        # ax.text(angle2rads(304), max_kg*2.31, sottotitolo.replace("1 mesi", "1 mese"), color="0.3", fontsize=11, ha="left")
        ax.text(angle2rads(304), max_kg*2.31, sottotitolo, color="0.3", fontsize=11, ha="left")

    fig.tight_layout(pad=3)  # fa in modo che i grafici non siano appiccicati

    fname = "tempoeranei/Grafico_schedaWorkout.png"
    plt.savefig(fname, dpi=300)
    img = open(fname, "rb")

    return img
