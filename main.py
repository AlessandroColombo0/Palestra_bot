import telebot
from telebot import types
from funzioni import weekday_nday_month_year, schedaFname_to_title, ora_EU, md_v2_replace, secs_to_hours, ic, is_digit, stringaPeso_to_num

# API

# telegram bot starting
# 0: tempo  1: nome  2: reps  3: serie  4: peso  5: note
TOKEN = "5012282762:AAEew28I_wMj6SJUWs7980BJr_LCXGree7k"
bot = telebot.TeleBot(TOKEN)
a_uid = 305444830
blank_char = "⠀"

# Bot started
ora = ora_EU(soloOre=True)  # HH:mm:ss
print(f"BOT STARTED {ora}")
keyboard_start = types.ReplyKeyboardMarkup(one_time_keyboard=False)
keyboard_start.add("/start", "/excel", "/tempi", "/database", "/archivio", "/score")

# ! disclaimer keyboards: nell'api dei bot di telegram i messaggi che hanno un reply_markup non possono essere modificati, inoltre l'unico modo per mostrare un reply_markup è
    # mandando un messaggio che la contiene
# messaggio vuoto per la keyboard
bot.send_message(chat_id=a_uid, text=blank_char+"🟢", disable_notification=True, reply_markup=keyboard_start)

botStarted_dict = {"txt": f"`{ora}` *\>* Bot started\n"}
botStarted_dict["msg"] = bot.send_message(chat_id=a_uid, text=botStarted_dict["txt"], disable_notification=True, parse_mode="MarkdownV2")


def addTo_botStarted(new_txt):
    ora = ora_EU(soloOre=True)
    botStarted_dict["txt"] += f"`{ora}` *\>* {new_txt}\n"
    bot.edit_message_text(text=botStarted_dict["txt"], chat_id=a_uid, message_id=botStarted_dict["msg"].message_id, parse_mode="MarkdownV2")

    print(f"\n{botStarted_dict['txt']}\n")

def bot_trySend(msg_text, reply_markup=None, parse_mode="MarkdownV2"):
    try:
        msg_ = bot.send_message(a_uid, msg_text, parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as exc:
        msg_ = bot.send_message(a_uid, msg_text+f"\nMarkdown error\n{exc}", parse_mode=None, reply_markup=reply_markup)

    return msg_

from creazione_immagini import Grafico_peso, Grafico_4sett, Grafico_mensile, schedaIndex_to_image, Grafico_radarScheda
# from clone_db import clone_database

import copy

import threading

import time
import datetime as dt
import calendar

import json

# from os import system

import traceback
import sys
import os
import shutil

from pushover import Client

# from replit import db
from keep_alive import keep_alive

keep_alive()


# push notifier
client = Client("u1rztsuw5cguaeyb1y8baimqb2pw4g", api_token="a2sr9qhf5x6t2q9vfhhervbrfhq651")
# nota: la versione 1.3 non funzionava (diceva che la richiesta per mandarenotifiche era malformata) quindi ho scaricato la 1.2.2


# VARIBILI  #

nomi_in_scheda = {"pettobi": "Petto Bicipiti", "schiena": "Schiena", "spalletri": "Spalle Tricipiti", "altro": "Gambe", "addominali": "Addominali"}
pre_constant = 2  # numero di secondi che vengono tolti alle pause di esercizio: sarebbe il tempo che ci si impiega ad accendere il telefono e schiacciare su /done

# COMANDI

# comandi generali che si possono fare sia durante l'allenamento che non
dict_comandi_gen = {
    "/start":           "Inizia l'allenamento / riavvia il bot. Il bot non ha bisogno di questo comando per entrare in funzione",
    "/exit":            "Serve per tornare nella sequences_usermode \"none\", ad esempio se abbiamo scritto /p e poi /exit è come se non avessimo mai fatto /p",
    "/sintassi":        "Ricevi il significato della sintassi delle ripetizioni e peso",
    "/excel":           "Ricevi il file di excel",
    "/tempi":           "Ricevi le previsioni sul tempo medio che si impiegherà per gli esercizio",
    "/database":        "Visualizza il database",
    "/info":            "Visualizza le informazioni su come dovrebbero essere il database, excel, le scritture di peso e altro",
    "/comandi":         "Visualizza i comandi",
    "/msgdebug":        "Togli il markdown a tutti i messagi, può essere utile se il bot ha problemi di markdown o per debug",
    "/nuovascheda":     "Cambia la scheda, quando si eseguirà questo comando verrà creato un nuovo diz_all nel db che dovrà essere compilato con /start, workout dopo workout",
    "/archivio":        "Visualizza le vecchie schede",
    "/score":           "Segna un esercizio come completato",
    "/peso":            "Ottieni il grafico del peso",
    "/forcetraining":   "Forza il training del modello, è utile quando il modello non è riuscito ad allenarsi quindi db[\"new training data\"] è vuoto ma il modello non si è allenato ed è ancora alla versione precedente. Negli altri casi potrebbe fare più danni che altro",
    "NN.NN / NN,NN":    "Segna il peso della bilancia, N sta per numero"
}

# comandi eseguibili solo durante allenamento
dict_comandi_dur = {
    "/done":    "Fai partire il timer per il prossimo esercizio",
    "/p":       "Modifica peso esercizio",
    "/ese":     "Modifica nome esercizio",
    "/rep":     "Modifica rep esercizio",
    "/n":       "Modifica note",
    "/notime":  "Non far segnare al bot il tempo che ci metterai a finire questo workout",
    "/sk":      "Skippa esercizio",
    "/back":    "counter -= 1, torna indietro di un esercizio",
    "/sw":      "Vai  a un'altro esercizio",
    "/fine":    "All'ultima serie dell'ultimo esercizio durante la compilazione scheda. prima di farlo assicurarsi che tutte le modifiche necessarie siano state effettuate. Se "
                "si fa dopo aver fato /done per sbaglio funziona lo stesso senza problemi"
}

all_commands_dict = copy.deepcopy(dict_comandi_gen)
all_commands_dict.update(dict_comandi_dur)
all_commands_dict = {k: k for k in all_commands_dict}


# Manipolazione database #

def db_insert(indexers, val, mode="="):
    if type(indexers) != list:
        indexers = [indexers]
    err = None
    v = db

    for i, indexer in enumerate(indexers):
        if type(v) == dict:
            if indexer in v:
                v = v[indexer]
            else:
                err = f"KeyError: {indexer} isn't a key in {v}"
                break
        elif type(v) == list:
            if type(indexer) == str:
                err = f"IndexError: using a string as a list index (idx: {indexer}, list: {v})"
                break
            elif indexer < len(v):
                v = v[indexer]
            else:
                err = f"IndexError: {indexer}th index is out of range in {v}"
                break
        else:
            err = f"TypeError: {indexer} was used as index for {v}"
            break

    if err:
        err = "Error in db_insert function:\n" + err
        bot_trySend(msg_text=err)
    else:
        indexing_str = "".join([f"['{i}']" if type(i) == str else f"[{i}]" for i in indexers])
        if mode != "append":
            exec(f"db{indexing_str} {mode} val",
                 {"db": db, "val": val})
        else:
            exec(f"db{indexing_str}.append(val)",
                 {"db": db, "val": val})

        with open(f"{db_file}", "w") as file:
            json.dump(db, file, indent=4)

def preserve_msg(msg):
    db_insert("preserved_msgIds", msg.message_id, mode="append")


actual_environment = True  # questa variabile è True se stiamo usando il bot da pythonanywhere e False se lo stiamo facendo da pycharm
# db["schede"]["2023-06-06"]["schiena"][1][3] = 4

import psutil
import platform
# pip install py-cpuinfo
import cpuinfo
import socket

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

uname = platform.uname()
cpufreq = psutil.cpu_freq()
svmem = psutil.virtual_memory()
sys_info_msg = f"""
<b>System Information</b>
System: {uname.system}
Node Name: {uname.node}
Release: {uname.release}
Version: {uname.version}
Machine: {uname.machine}
Processor: {uname.processor}
Processor: {cpuinfo.get_cpu_info()['brand_raw']}
Ip-Address: {socket.gethostbyname(socket.gethostname())}

<b>CPU Info</b>
Physical cores: {psutil.cpu_count(logical=False)}
Total cores: {psutil.cpu_count(logical=True)}
Max Frequency: {cpufreq.max:.2f}Mhz
Min Frequency: {cpufreq.min:.2f}Mhz
Current Frequency: {cpufreq.current:.2f}Mhz
Total CPU Usage: {psutil.cpu_percent()}%

"<b>Memory Information</b>
Total: {get_size(svmem.total)}
Available: {get_size(svmem.available)}
Used: {get_size(svmem.used)}
Percentage: {svmem.percent}%
"""

if uname.node == "iMac-2.local":
    actual_environment = False
    db_file = "db_test.json"
    addTo_botStarted("Running in *test* environment")
else:
    actual_environment = True
    db_file = "db.json"
    addTo_botStarted("Running in *real* environment")

db = json.load(open(db_file, "r"))
startup_db = copy.deepcopy(db)

bot_trySend(sys_info_msg, parse_mode="HTML")
end_sleepTime = 40 if actual_environment else 5  # secondi di sleep alla fine dell'allenamento

key = "aumenti_msgScheda"
if key not in db.keys():
    db[key] = {"addominali": [], "pettobi": [], "schiena": [], "spalletri": [], "altro": []}


# DICHIARAZIONI

nomi_workout = ["Addominali", "Petto Bi", "Schiena", "Spalle Tri", "Altro"]
lista_es_simple = ["addominali", "pettobi", "schiena", "spalletri", "altro"]
start_dict = {"Addominali": "addominali", "Petto Bi": "pettobi", "Schiena": "schiena", "Spalle Tri": "spalletri", "Altro": "altro"}
esercizi_title_dict = {"addominali": "Addominali", "pettobi": "Petto Bicipiti", "schiena": "Schiena", "spalletri": "Spalle Tricipiti", "altro": "Altro"}

serie_in_ese_list, count_serie, counter_dict, flat_timer_msg_list, flat_pause_list = [{} for _ in range(5)]

m_to_mesi = {"01": "Gen", "02": "Feb", "03": "Mar", "04": "Apr", "05": "Mag", "06": "Giu", "07": "Lug", "08": "Ago", "09": "Set", "10": "Ott", "11": "Nov", "12": "Dic"}

global_dict = {"counter": 0, "switch_msg": 0, "inizio_allenamento": "", "inizio": 0.0, "inizio_str": "", "continuo": 0.0, "continuo_n2": 0.0, "numero_es_cambiato": 0,
               "start_message_id": 0, "msg_scheda": 0, "msg_fine-timer": 0, "creazione_scheda": False}
levers_dict = dict(timer=False, start_sw=True, switch_first=True, notime=False, restart=False, grafico_peso=False, radar_graph=False, skip=False)

usermode = {a_uid: "none"}
sequences_usermode = {a_uid: "none"}

strikethrough_list_msg = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" "", "", ""]

order = [-1, 0, 1]

# markdown
pm = {"mdV2": "MarkdownV2", "HTML": "HTML", "md": "Markdown"}
pm_toggle = False

workouts_lists = {"addominali": [], "pettobi": [], "schiena": [], "spalletri": [], "altro": []}
switch_list, switch_list_quotes, counter_dict_timer_full, prev_tempo_building, peso_ese_list = [copy.deepcopy(workouts_lists) for _ in range(5)]
""" prev_tempo_building: andrà a sostituire il prev_tempo più vecchio, che a sua volta era un prev_tempo_building, db["prev_tempo"] è formato da 3db db_prev_tempo building """

# counter
count_serie_flat = {}
completed_counters = {"addominali": [0], "pettobi": [0], "schiena": [0], "spalletri": [0], "altro": [0]}

# creazione scheda
strutturaBase_dizAll = [[60, f"{i+1}° Esercizio", "/", 4, "/", False] for i in range(20)]  # 20 esercizi vuoti, 60 pausa di base e 4 serie di base, c'è lo stretto necessario
    # per far si che funzioni in un allenamento. False serve per indicare alla creazione messaggio scheda che è una struttura base e non un vero esercizio

creazioneScheda_undo = {}
creazioneScheda_undo["cambioScheda_msgId"] = None  # inizializziamo con None perchè ci serve per far sapere al bot se sono già stati mandati dei messaggi di compilazione o meno

type_to_dizAll_idx = {"Nome": 1, "Reps": 2, "Peso": 4, "Note": 5}

str_0_9_list = [str(i) for i in range(0,10)]
str_1_6_list = [str(i) for i in range(1,7)]
int_to_type = {0: "Nome", 1: "Reps", 2: "Peso", 3: "Note"}
dizAllIdx_to_type = {0: "Pausa", 1: "Nome", 2: "Reps", 3: "Serie", 4: "Peso", 5: "Note"}
letter_to_type = {"E": "Nome", "R": "Reps", "P": "Peso", "N": "Note"}


# KEYBOARDS
keyboard_workouts = types.ReplyKeyboardMarkup(one_time_keyboard=True)
keyboard_done = types.ReplyKeyboardMarkup(one_time_keyboard=False)

# peso
keyb_addominali, keyb_pettobi, keyb_schiena, keyb_spalletri, keyb_altro = [copy.deepcopy(keyboard_workouts) for _ in range(5)]
# switch
keyb_addominali_sw, keyb_pettobi_sw, keyb_schiena_sw, keyb_spalletri_sw, keyb_altro_sw = [copy.deepcopy(keyboard_workouts) for _ in range(5)]
# note / ese
keyb_addominali_n, keyb_pettobi_n, keyb_schiena_n, keyb_spalletri_n, keyb_altro_n = [copy.deepcopy(keyboard_workouts) for _ in range(5)]

keyb_peso = {"addominali": keyb_addominali, "pettobi": keyb_pettobi, "schiena": keyb_schiena, "spalletri": keyb_spalletri, "altro": keyb_altro}
keyb_names = {"addominali": keyb_addominali_n, "pettobi": keyb_pettobi_n, "schiena": keyb_schiena_n, "spalletri": keyb_spalletri_n, "altro": keyb_altro}

[keyboard_workouts.add(i) for i in nomi_workout]
keyboard_done.row("/done")
keyboard_done.row("/sw", "/p", "/ese", "/n")


# time prediction data di questo workout, verrà aggiunta al db
TP_data = dict(timedate=None, workout=None, n_workout=0,  # info
                S_nomi=[], S_reps=[], S_pause=[], S_serie=[],  # scheda
                ultimi_allenamenti=[], current=[])  # current = w_sec_taken di questo workout
# OBIETTIVO: PREVEDERE I SECONDI RESTANTI


# SCHEDA E DIZ ALL

def get_dataSchedaAttuale():
    date_schede = list(db["schede"].keys())
    date_schede.sort()
    return date_schede[-1]


prev_tempo = {}
for w in workouts_lists.keys():
    prev_tempo[w] = startup_db["w_sec_taken"][w][:3]

    diff = len(prev_tempo[w]) - 3
    if diff < 0 and prev_tempo[w]:
        [prev_tempo[w].append(prev_tempo[w][-1]) for i in range(abs(diff))]


avg_adj_prev_tempo = copy.deepcopy(workouts_lists)   # avg: è la media degli ultimi 3 allenamenti per ogni esercizio, adj: adjusted: quando facciamo un /done viene inserito in [ese] il
    # tempo che ci abbiamo messo, quindi usa il tempo che ci abbiamo messo in questo allenamento per dirci quanto tempo abbiamo messo dall'inizio, e per la previsione usa la media
    # degli scorsi allenamenti

global_dict["data_scheda_attuale"] = get_dataSchedaAttuale()
diz_all = startup_db["schede"][global_dict["data_scheda_attuale"]]

# FUNZIONI DIZ ALL ECC

def escape_dizAll(diz_all_):
    """ otteniamo un dizall dove i nomi ese, reps, peso e note sono compatibili con markdown v2 """
    for workout, lists in diz_all_.items():
        for i, list_ in enumerate(lists):
            diz_all_[workout][i][1] = md_v2_replace(diz_all_[workout][i][1])
            diz_all_[workout][i][2] = md_v2_replace(diz_all_[workout][i][2])
            diz_all_[workout][i][4] = md_v2_replace(diz_all_[workout][i][4])
            diz_all_[workout][i][5] = md_v2_replace(diz_all_[workout][i][5])

    return diz_all_

esc_dizAll = escape_dizAll(copy.deepcopy(diz_all))

def eseSerie_to_timerMsg(workout, ese, serie):
    quote = '"'  # usando \" replit aveva problemi nell'interpetazione del codice per qualche motivo
    timer_msg = f"_{esc_dizAll[workout][ese][1]}_\n" \
                f"*{serie + 1} serie su {esc_dizAll[workout][ese][3]}*\n" \
                f"*R:* {esc_dizAll[workout][ese][2]}\n" \
                f"*P:* {esc_dizAll[workout][ese][4]}\n" \
                f"*Pausa:* `{esc_dizAll[workout][ese][0]}{quote}`" \
                f"\nNote: _{esc_dizAll[workout][ese][5]}_".replace("\nNote: _None_", "").replace("\nNote: _False_", "")

    return timer_msg


def adapt_to_newSerie(workout):
    """ questa funzioe viene chiamata nella creazione iniziale dei dizionari e quando il numero di serie cambia e ricrea alcuni dizionari usando le serie nuove """
    global serie_in_ese_list
    global count_serie
    global counter_dict
    global flat_timer_msg_list
    global flat_pause_list

    serie_in_ese_list[workout] = [[0 for serie in range(diz_all[workout][ese][3])] for ese in range(len(diz_all[workout]))]
    """ serie_in_ese_list = {"addominali": [[serie_es-1, serie_es-1, serie_es-1], [serie_es-2, ...] ...] """

    counter_dict[workout] = []
    count_serie[workout] = [[] for _ in serie_in_ese_list[workout]]
    """
    count_serie[workout] = [[0, 1, 2, 3], [4, 5, 6, 7, 8], ...]  # cambia solo con /switch   
    
    prima del primo /done:  count_serie_flat[workout] =    [0, 1, 2, 3, 4, ...]
    dopo il primo /done:    count_serie_flat[workout] =    [1, 2, 3, 4, ...]
    dopo il secondo /done:  count_serie_flat[workout] =    [2, 3, 4, ...]
    """
    flat_timer_msg_list[workout] = []
    flat_pause_list[workout] = []

    serie_flat_counter = 0  # 0 1 2 3 4 5 6 7 ...

    for ese in range(len(serie_in_ese_list[workout])):
        counter_dict[workout].append(sum([diz_all[workout][i][3] for i in range(ese+1)]))
        """ counter_dict = {'addominali': [4, 8, 11], 'pettobi': [...], ...}, vuol dire che addominali aveva il  primo ese con 4 serie, il secondo con 4, terzo con 3 """

        for serie in range(len(serie_in_ese_list[workout][ese])):
            flat_timer_msg_list[workout].append(eseSerie_to_timerMsg(workout, ese, serie))

            flat_pause_list[workout].append(diz_all[workout][ese][0])

            count_serie[workout][ese].append(serie_flat_counter)
            serie_flat_counter += 1


def inizializzazione(workout):
    adapt_to_newSerie(workout)
    serie_count = 0

    # itera per ogni esercizio
    for ese in range(len(serie_in_ese_list[workout])):

        # creazione delle keyboard
        peso_ese_list[workout].append(f"{esc_dizAll[workout][ese][4]}  {esc_dizAll[workout][ese][1]}")  # peso e nome
        switch_list[workout].append(esc_dizAll[workout][ese][1])
        switch_list_quotes[workout].append("")

        # itera ogni serie
        for serie in range(len(serie_in_ese_list[workout][ese])):
            prev_tempo_building[workout].append(0)  # il prev tempo è formato da 0 che non vengono usati e gli idxs man mano vengono cambiati con il tempo impiegto

            if prev_tempo[workout]:  # se non è []
                # serve in caso ci fossero errori tra prev_tempo nel db e numero di serie totali, di norma non dovrebbe succedere ma è successo ed è un easy fix
                if len(db["w_sec_taken"][workout][0]) < serie_count+1:
                    [db_insert(["w_sec_taken", workout, i], 100, mode="append") for i in range(len(db["w_sec_taken"][workout]))]

                avg_adj_prev_tempo[workout].append(sum([prev_tempo[workout][i][serie_count] for i in range(3)]) / 3)

            serie_count += 1


        keyb_names[workout].add(esc_dizAll[workout][ese][1])

    [keyb_peso[workout].add(i) for i in peso_ese_list[workout]]

active_workouts = [workout for workout, excercises in diz_all.items() if excercises != [] or excercises != strutturaBase_dizAll]

for gruppo_muscolare in active_workouts:
    inizializzazione(gruppo_muscolare)  # questo vuol dire che creiamo dizionari e liste solo per i gruppi muscolari che esistono nella scheda

switch_list_full = copy.deepcopy(switch_list)

# modifiche a diz_all
modifiche_dict = {
    "/p": {"msg0": "Scegli l'esercizio a cui vuoi cambiare il peso",
           "kb0": keyb_peso,
           "iter_list": peso_ese_list,
           "db_idx": 4,
           "label": "peso"},
    "/ese": {"msg0": "Scegli il nome dell'esercizio da modificare",
             "kb0": keyb_names,
             "iter_list": switch_list_full,
             "db_idx": 1,
             "label": "nomi"},
    "/n": {"msg0": "Scegli l'esercizio a cui vuoi cambiare le note",
           "kb0": keyb_names,
           "iter_list": switch_list_full,
           "db_idx": 5,
           "label": "note"},
    "/rep": {"msg0": "Scegli l'esercizio a cui vuoi cambiare le ripetizioni",
             "kb0": keyb_names,
             "iter_list": switch_list_full,
             "db_idx": 2,
             "label": "reps"},
}


"""FUNZIONI"""

# FUNZIONI TEMPO

def secs_to_time(seconds, addo_bool):
    """
    funziona che ci dice a che ora finiremo l'allenamento in base a quando abbiamo iniziato e dopo quanti secondi è previsto che finiamo
    ci dice anche quanti minuti mancano all'ora che viene indicata come fine dell'allenamento
    """
    # fine = ora in cui finiremo l'allenamento in secondi, è formato dai secondi che corrispondono all'ora di quando abbiamo iniziato l'esercizio + i secondi che ci metteremo a
        # finire (secondo le previsioni)
    fine = sum([a * b for a, b in
                zip([3600, 60, 1], map(int, global_dict["inizioAll_HH:MM:SS"].split(":")))]) + seconds

    ora = ora_EU(soloOre=True)
    ora_in_secondi = sum([a*b for a,b in zip([3600,60,1], map(int,ora.split(":")))])

    if addo_bool == False:
                    #  HH:MM:SS                                 # - 1h 13m
        return f"`{str(dt.timedelta(seconds=fine))[0:5]}`, *\-{secs_to_hours(abs(fine - ora_in_secondi))}*"
    else:
        return f"`{str(dt.timedelta(seconds= fine))[0:5]}`, *\-{secs_to_hours(abs(fine - ora_in_secondi))}*\n" \
               f"🧭 Con addominali: `{str(dt.timedelta(seconds= fine + sum(avg_adj_prev_tempo['addominali'])))[0:5]}`, *\-{secs_to_hours(fine + sum(avg_adj_prev_tempo['addominali']) - ora_in_secondi)}*"

def get_hh_mm_ss(integer=False):
    hh, mm, ss = ora_EU(soloOre=True).split(":")

    if integer:
        ints = [int(i) for i in [hh, mm, ss]]
        hh, mm, ss = ints

    return hh, mm, ss

def numero_ora():
    h, m, s = get_hh_mm_ss(integer=True)

    return h*3600 + m*60 + s

def precise_numero_ora():
    """ questa funzione contiene anche i millisecondi e viene usata nel timer """
    ora = str(ora_EU())
    h, m, s = get_hh_mm_ss(integer=True)
    millisecs = float(ora[19:])

    return h*3600 + m*60 + s + millisecs


# FUNZIONI ALLENAMENTO

def apply_modification(text, workout, i_ese, i_column, creazione_scheda=False):
    """ questa funzione viene usata quando si modifica qualcosa nel db, ad esempio quando si fa /p o si stra creando una nuova scheda. quello che fa è attuare le modifiche nel
    db, in diz_all, esc_dizall e in flat_timer_msg """
    # se sono peso cambia da k a kg, se sono nomi maiuscola all'inizio
    text = text.replace("k", "㎏") if i_column == 4 else text
    text = text[0].upper() + text[1:] if i_column == 1 else text  # .capitalize() toglieva l'upper da tutto e lo metteva solo all'inizio

    if not creazione_scheda:  # se siamo in fase di creaizone scheda non viene applicata al database
        db_insert(["schede", global_dict["data_scheda_attuale"], workout, i_ese, i_column], text)
        # db["schede"][global_dict["data_scheda_attuale"]][workout][i_ese][i_column] = text

    if creazione_scheda:
        creazioneScheda_undo["previous_val"] = diz_all[workout][i_ese][i_column]  # salviamo il testo che c'era prima, in questo modo in caso di errori possiamo facilmente
            # rimetterlo usando questa stessa funzione

    diz_all[workout][i_ese][i_column] = text
    esc_dizAll[workout][i_ese][i_column] = md_v2_replace(text)  # modifichiamo l'esc_dizAll in modo che vengano modificati anche i messaggi scheda

    # ricostruzione timer msg, usa esc_dizAll
    counter_massimo = counter_dict[workout][i_ese]
    counter_minimo = counter_dict[workout][i_ese-1] if i_ese != 0 else 0  # è 0 se è il primo esercizio
        # visto che counter 0 = primo esercizio prima serie, altrimenti è il limite massimo dell'es prima
    for i_serie, counter in enumerate(range(counter_minimo, counter_massimo)):  # counter: counter del timer_msg da mettere nella flat list, i_serie: serve per fare N serie su N
        flat_timer_msg_list[workout][counter] = eseSerie_to_timerMsg(workout, i_ese, i_serie)

def cambioScheda_compile(text, workout, i_ese, dizAll_idx, uid, forced_reps=False, change_buttons=None, bot_probs=None, callback=False, CB_new_dizAll_idx=None, CB_new_val=None):
    mk = ">" if not callback or CB_new_dizAll_idx else "╳"
    type = dizAllIdx_to_type[dizAll_idx]

    if dizAll_idx == 3:  # serie
        diz_all[workout][i_ese][3] = int(text)
        esc_dizAll[workout][i_ese][3] = str(text)
        adapt_to_newSerie(workout)

    elif dizAll_idx == 0:  # pausa
        apply_modification(int(text[:-1]), workout, i_ese, 0, True)
        global flat_pause_list
        flat_pause_list[workout] = [esc_dizAll[workout][ese][0] for ese in range(len(serie_in_ese_list[workout]))
                                    for serie in range(diz_all[workout][ese][3])]

    elif forced_reps:  # caso in cui è stato scritto "N reps"
        apply_modification(text[:-5], workout, i_ese, 2, True)

    elif dizAll_idx in [1, 2, 4, 5]:
        # applicazione e messaggio
        apply_modification(text, workout, i_ese, dizAll_idx, True)
        if CB_new_dizAll_idx:
            apply_modification(CB_new_val, workout, i_ese, CB_new_dizAll_idx, True)

    txt_scheda, _ = crea_messaggioScheda(workout, send=False)
    try:
        bot.edit_message_text(text=txt_scheda, chat_id=uid, message_id=global_dict["msg_scheda"], parse_mode=pm["mdV2"])
    except:
        pass

    if not callback and not CB_new_dizAll_idx:
        text = f"{md_v2_replace(diz_all[workout][i_ese][dizAll_idx])}` {mk} `*{type}*"
    else:
        if CB_new_dizAll_idx:
            type = dizAllIdx_to_type[CB_new_dizAll_idx]
        text = f"{CB_new_val}` {mk} `*{type}*"
    if dizAll_idx in [1, 2, 4, 5] and not callback:
        text += f"\n🤖:` {bot_probs}`"

    msg_ = bot_trySend(text, change_buttons)

    return msg_

def messaggio_scheda(workout):
    msg_scheda_tot = f"<code>{blank_char}    </code><b>{nomi_in_scheda[workout].upper()}</b>\n\n"
    circlepos = 0
    circle_lever = True

    cloned_aum_msgScheda = [{k: v for k, v in aum_dict.items()} for aum_dict in db["aumenti_msgScheda"][workout]]
    aums = [i | {"w_distance": startup_db["workout_count"][workout] - i["w_count"]} for i in cloned_aum_msgScheda]
    aums = [i for i in aums if i["w_distance"] < 3]  # 0: questo workout, 1: scorso workout, 2: workout prima dello scorso

    ese_idxs = [i["ese_idx"] for i in aums]
    aums_dict = {idx: [] for idx in set(ese_idxs)}
    for i in aums:
        aums_dict[i["ese_idx"]].append(i)
    aums_dict = {idx: i[-1] for idx, i in aums_dict.items()}  # l'ultimo item dovrebbe è il più recente, quindi lo scegliamo

    for ese in range(len(counter_dict[workout])):
        # strikethrough
        if switch_list[workout][ese] == "":  # se l'esercizio è già stato completato
            strikethrough_list_msg[ese] = "<strike>"

        if global_dict["counter"] < counter_dict[workout][ese] and circle_lever == True:
            circlepos = ese  # int che indica a quale esercizio mettere il cerchio
            circle_lever = False  # una volta trovato non ci ritorna su questo punto di codice

    for ese in range(len(serie_in_ese_list[workout])):
        # ... in caso di creazione scheda
        if ese == 0:  # serve per evitare di fare il prossimo if statement se l'index è 0 (ci sarebbe index error)
            pass
        elif diz_all[workout][ese-1][5] != False:  # se le note sono False, quindi se è una struttura base db
            pass
        else:  # se l'esercizio prima di questo era una struttura base
            msg_scheda_tot += "<code>    </code>..."
            break

        # questa parte di codice non viene raggiunta se si finisce nell'else perchè c'è il break
        pausa = str(diz_all[workout][ese][0])
        pausa_padding = "  " if len(pausa) == 2 else " "  # questo codice serve per padding (mantenere la lunghezza uguale anche quando ci sono 3 numeri al posto che 2)
        pau = f'{pausa}"{pausa_padding}'

        nome_ese = diz_all[workout][ese][1]
        circle = "⚪️️️ " if ese == circlepos else ""
        slm1 = strikethrough_list_msg[ese]
        slm2 = f"{strikethrough_list_msg[ese][:1]}/{strikethrough_list_msg[ese][1:]}" if slm1 == "<strike>" else ""

        aum_old = ""
        aum_emoji = " "*5
        if ese in aums_dict:
            emojis = {0: "💹", 1: "✳️", 2: "*️⃣"}
            aum_emoji = f"  {emojis[aums_dict[ese]['w_distance']]} "
            aum_old = f"\n<code>     </code><strike>{aums_dict[ese]['peso']}</strike>"

        msg_scheda_tot += f"<code>{pau}</code>" +       f"{circle}{slm1}<i>{nome_ese}</i>{slm2}\n" \
                          f"<code>     </code>" +       f"S: {diz_all[workout][ese][3]}   R: {diz_all[workout][ese][2]}\n" \
                          f"<code>{aum_emoji}</code>" + f"{diz_all[workout][ese][4]}{aum_old}\n\n"

    return msg_scheda_tot

def crea_messaggioScheda(workout, send=False):

    addo_bool = False if workout == "addominali" else True  # ci viene detto quanto ci mettiamo a fare anche gli addominali solo se stiamo facendo un esercizio diverso da addominali
    fine_prevista = f"🧭 Fine prevista: {secs_to_time(sum(avg_adj_prev_tempo[workout]), addo_bool)}" if global_dict["creazione_scheda"] == False else ""

    msg_scheda = messaggio_scheda(workout)
    msg_schedaPrev = fine_prevista
    if send:
        # questi due messaggi devono essere separati, visto che solo msg_scheda subisce degli edit_message, e ci serve un messaggio che non viene modificato per usare le keyboards
        msg_scheda = bot_trySend(msg_scheda, parse_mode="HTML")
        if fine_prevista != "":
            msg_schedaPrev = bot_trySend(msg_schedaPrev, reply_markup=keyboard_done)

    return msg_scheda, msg_schedaPrev

def fine_allenamento(workout, uid, score=False):
    """ funzione che si occupa di fare tutte le scritture nel db e del messaggio alla fine di un allenamento """
    ora_eu_iso = ora_EU(isoformat=True)
    fine_str = ora_EU(soloOre=True)

    # messaggio allenamento finito e aggiunta di questa fine allenamento nel db
    msg_fine = ""
    for i, fine_es in enumerate(reversed(db["fine_allenamenti"][workout])):
        # primi 3 esercizi passati
        if i != 0:
            msg_fine += fine_es + "\n\n"
        else:
            pass

    # caso in cui c'è stato un allenamento con /start /done fino alla fine
    if not score:
        # msg_oggi
        fine = time.perf_counter()
        tempo_imp = secs_to_hours(fine-global_dict["inizioAll_secs"], False)

        msg_oggi = f"Tempo impiegato: *{tempo_imp}*\nDalle {global_dict['inizioAll_HH:MM:SS'][:-3]} alle {fine_str[:-3]}"

        # se il notime è attivato salta il passaggio altrimenti salva il tempo impiegato
        if levers_dict["notime"] == False:
            # scrittura del prev_tempo nel db
            global_dict["continuo_n2"] = time.perf_counter()
            prev_tempo_building[workout][global_dict["counter"]] = prev_tempo_building[workout][global_dict["counter"] - 1]  # facciamo in modo che la durata dell'ultimo
                # esercizio sia uguale a quella del penultimo
            prev_tempo_building[workout] = [i for i in prev_tempo_building[workout] if i != 0]

            db_insert(["w_sec_taken", workout], prev_tempo_building[workout], mode="append")
            # db["w_sec_taken"][workout].append(prev_tempo_building[workout])

        else:  # se invece abbiamo usato /notime lo scriviamo nel msg_oggi
            msg_oggi += " \(/notime\)"

    else:
        msg_oggi = f"Tempo impiegato: *N/A*\nFine: {fine_str}"

    # parte di saving dizall 2 volte e conteggio workout
    db_insert(["workout_count", workout], 1, mode="+=")
    # db["workout_count"][workout] += 1

    if db["workout_count"][workout] == 2:
        db_insert(["data_2volte", workout], ora_eu_iso)
        # db["data_2volte"][workout] = ora_eu_iso

        db_insert(["dizAll_2volte", workout],
                  copy.deepcopy(db["schede"][global_dict["data_scheda_attuale"]][workout]))

    # msg
    msg_ = bot.send_message(chat_id=uid, text= f"🔴 Allenamento finito: *{esercizi_title_dict[workout]}:* \({db['workout_count'][workout]}ª volta\)\n\n"
                                               f"{msg_fine}`Oggi`\n{msg_oggi}",
                            reply_markup=keyboard_start, parse_mode=pm["mdV2"])
    db_insert("preserved_msgIds", msg_.message_id, mode="append")
    # db["preserved_msgIds"].append(msg_.message_id)

    # eliminazione messaggi
    for messaggio in range(global_dict["start_message_id"], msg_.message_id):
        bot_tryDelete(uid, messaggio)

    db_insert(["fine_allenamenti", workout], [db["fine_allenamenti"][workout][i] for i in [-1,0,1,2]])
    str_date = f"{int(ora_eu_iso[8:10])} {m_to_mesi[ora_eu_iso[5:7]]}"
    db_insert(["fine_allenamenti", workout, 0], f"`{str_date}`\n{msg_oggi}")
    # db["fine_allenamenti"][workout][0] = f"`{str_date}`\n{msg_oggi}"

    # ultimo allenamento
    db_insert("allenamenti", [ora_EU(isoformat=True), workout], mode="append")

    global_dict["creazione_scheda"] = True
    sequences_usermode[uid] = "none"

def saveSend_TP(workout):
    TP_data["current"] = prev_tempo_building[workout]
    db_insert("TP_data", TP_data, mode="append")
    # db["TP_data"].append(TP_data)

    fname = "temporanei/tp_data.json"
    with open(fname, "w") as file:
        json.dump(TP_data, file, indent=4)

    bot_trySend(f"<b>! TP data !</b>", parse_mode=pm["HTML"])
    bot.send_document(a_uid, open(fname, "rb"))

def done(msg, workout):
    uid, msg_id = msg.chat.id, msg.message_id

    if sequences_usermode[uid] != "none":
        sequences_usermode[uid] = "none"
        bot_tryDelete(uid, msg_id-1)

    bot_tryDelete(uid, msg_id)
    bot_tryDelete(uid, global_dict["msg_scheda"])
    if global_dict["counter"] != 0 and not global_dict["creazione_scheda"]:
        bot_tryDelete(uid, global_dict["msg_schedaPrev"])
    bot_tryDelete(uid, global_dict["msg_fine-timer"])

    # PREV TEMPO

    # primo /done
    if global_dict["counter"] == 0:
        global_dict["continuo"] = time.perf_counter()
        prev_tempo_building[workout][global_dict["counter"]] = global_dict["continuo"]-global_dict["inizioAll_secs"]

    # tutti i /done dopo il primo
    else:
        global_dict["continuo_n2"] = time.perf_counter()
        prev_tempo_building[workout][global_dict["counter"]] = global_dict["continuo_n2"]-global_dict["continuo"]
        global_dict["continuo"] = time.perf_counter()

    if len(avg_adj_prev_tempo[workout]) > global_dict["counter"]:  # condizione ordinaria
        avg_adj_prev_tempo[workout][global_dict["counter"]] = prev_tempo_building[workout][global_dict["counter"]]
    else:  # questo caso serve ad evitare bug nel caso in cui avg_adj_prev_tempo è più piccolo del prev tempo build
        avg_adj_prev_tempo[workout].append(prev_tempo_building[workout][global_dict["counter"]])

    t = threading.Thread(target=Timer, args=[msg, workout])
    t.start()


# FUNZIONI GENERICHE

def send_db_json():
    if actual_environment:
        date = weekday_nday_month_year(ora_EU(), year=True)
        fname = f"db/backup-db_{date}.json"
        with open(fname, "w") as file:
            json.dump(db, file, indent=4)

        sendable_cloned_db = open(fname)
        msg_ = bot.send_document(a_uid, sendable_cloned_db)
        preserve_msg(msg_)
        # db_insert("preserved_msgIds", msg_.message_id, mode="append")
        # db["preserved_msgIds"].append(msg_.message_id)

        return open(fname)

def somma_numeri_in_stringa(stringa):
    # queste operazioni di replacement ci servono per trattare in modo diverso i ", " e ",", visto che una separa il peso e l'altra è un numero con la virgola
    stringa = stringa.replace(", ", "$$$")
    stringa = stringa.replace(",", ".")
    stringa = stringa.replace("$$$", ", ")

    num_string = ""
    num_lever = False

    sum_list = []

    for char in stringa:
        # se il carattere è un numero
        if char in ["0","1","2","3","4","5","6","7","8","9",".","-"]:
            num_string += char
            num_lever = True

        # se il carattere precedente era un numero
        elif num_lever:
            num_string += " "
            num_lever = False

        # se il carattere precedente non era un numero e questo carattere non è un numero
        else:
            pass

    num_string = num_string.replace("- ", "-")
    for numero in num_string.split(" "):
        if numero != "":
            try:
                sum_list.append(float(numero))
            except:
                pass

    return sum(sum_list)

def reverse_dict(dict):
    keys_list = list(dict.keys())
    values_list = list(dict.values())

    return {values_list[i]: keys_list[i] for i in range(len(keys_list))}

def flatten_list(lista):
    return [i for sublist in lista for i in sublist]


# FUNZIONI TELEGRAM

def split_messaggio(uid, string, parse_mode=None):
    if len(string) > 4096:
        for message_limit in range(0, len(string), 4096):
            bot.send_message(uid, string[message_limit:message_limit+4096])
    else:
        bot.send_message(uid, string, parse_mode=parse_mode)

def inline_buttons_creator(raw_buttons, row_width):
    """
    {"texts", ["1","2","3"], "callbacks": ["01","02","03"]} -> keyboard(button(text="1", callback_data="01"), ...)
    """
    buttons_list = []
    for i, text in enumerate(raw_buttons["texts"]):
        buttons_list.append(types.InlineKeyboardButton(text=text, callback_data=raw_buttons["callbacks"][i]))

    keyboard = types.InlineKeyboardMarkup(row_width=row_width)
    keyboard.add(*buttons_list)

    return keyboard

def bot_tryDelete(uid, message_id):
    try:
        if message_id not in db["preserved_msgIds"]:
            bot.delete_message(uid, message_id)
    except:
        pass

def traceback_message(exc):
    bot.send_message(a_uid, f"ERRORE: {exc} \nRicordarsi di salvare il traceback se lo si vuole visualizzare in futuro")
    fname = "temporanei/latest_traceback.txt"
    with open(fname, 'w') as f:
        f.write(str(traceback.format_exc()))
    file = open(fname, "rb")
    msg_ = bot.send_document(a_uid, file)

    return msg_


# FUNZIONI AI

def idx_to_oneHot(idx, labels_num):
    one_hot = [0. for _ in range(labels_num-1)]
    one_hot.insert(idx, 1.)

def start_training_procedure():
    try:
        bot.send_message(a_uid, "Allenamento del modello in corso...")

        from training import train_model
        summary_msg, loaded_files = train_model("msg labelling", "hpo_model", db)

        bot.send_message(a_uid, summary_msg, parse_mode="HTML")
        [bot.send_document(a_uid, file, parse_mode=pm["mdV2"]) for file in loaded_files]

    except Exception as exc:
        msg_ = bot.send_message(a_uid, text="L'allenamento del modello non è andato a buon termine:")
        preserve_msg(msg_)
        msg_ = traceback_message(exc)
        preserve_msg(msg_)

# PRINT

print_ = False
if print_:
    startup2_elements = {"diz_all": diz_all, "serie_in_ese_list": serie_in_ese_list, "flat_timer_msg_list": flat_timer_msg_list, "flat_pause_list": flat_pause_list,
                         "switch_list": switch_list, "counter_dict": counter_dict, "count_serie": count_serie, "switch_list_quotes": switch_list_quotes,
                         "prev_tempo_building": prev_tempo_building, "avg_adj_prev_tempo": avg_adj_prev_tempo,
                         "keyb_pettobi_n": keyb_pettobi_n}
    for var_name, var in startup2_elements.items():
        padding = 25 - len(var_name)
        print(var_name + " "*padding + str(var))


# TIMER

def Timer(msg, workout):
    refresh_rate = 1
    num_ora_inizio = precise_numero_ora()
    lever_10_sec = True

    if switch_list[workout][0] == switch_list_full[workout][0] and levers_dict["switch_first"] == True:
        for ese in switch_list[workout]:
            if ese == "":
                completed_counters[workout].remove(0)
                levers_dict["switch_first"] = False
                break

    count_serie_flat[workout] = flatten_list(count_serie[workout])
    [count_serie_flat[workout].remove(counter) for counter in completed_counters[workout]]  # rimozione dei counter già completati

    # counter e index dell'esercizio a cui siamo
    global_dict["counter"] = count_serie_flat[workout][0]
    counter = global_dict["counter"]

    # ottenimento di i_ese
    for i_ese, max_counter in enumerate(([0] + counter_dict[workout])[1:]):  # inseriamo uno 0 all'inizio e partiamo dal secondo elemento per non avere problemi di iterazione
        if counter < max_counter and counter >= counter_dict[workout][i_ese-1]:  # se il counter è minore del counter massimo per l'esercizio e maggiore del coutner massimo di prima
            global_dict["i_ese"] = i_ese
            break

    # messaggio scheda
    msg_scheda, msg_schedaPrev = crea_messaggioScheda(workout, send=True)
    global_dict["msg_scheda"] = msg_scheda.message_id
    if type(msg_schedaPrev) != str:
        global_dict["msg_schedaPrev"] = msg_schedaPrev.message_id

    #messaggio con timer
    msg_timer = bot_trySend(f"{str(flat_timer_msg_list[workout][global_dict['counter']])} *{str(flat_pause_list[workout][global_dict['counter']])}*")
    global_dict["msg_timer"] = msg_timer.message_id

    refreshed_num_ora = 0
    secs_after = time.perf_counter()
    levers_dict["timer"] = True


    # DURANTE TUTTO L'ALLENAMENTO, FINO ALLA FINE
    while True:
        numOra_fineTimer = num_ora_inizio + flat_pause_list[workout][counter] - pre_constant

        # DURANTE TIMER
        if levers_dict["timer"] == True and refreshed_num_ora + refresh_rate < numOra_fineTimer and levers_dict["skip"] == False:

            refreshed_num_ora = precise_numero_ora()
            timer_piece = f"🟡 *Timer*: `{int(numOra_fineTimer-refreshed_num_ora)}\"`"

            # 10 SECONDS WARNING, se mancano 10 secondi:
            if lever_10_sec == True and refreshed_num_ora > numOra_fineTimer - 9:
                noti = flat_timer_msg_list[workout][global_dict['counter']].replace("/", "").replace("*", "").replace("_", "").replace("`", "")  # nelle notifiche non c'è markdown
                client.send_message(f"🟡 8 secondi restanti\n{noti}", title="Timer")
                lever_10_sec = False

            # TIMER MESSAGE EDITING, se siamo ancora < di numOra_fineTimer:
            if (refreshed_num_ora + refresh_rate + 0.1) < numOra_fineTimer:

                try:
                    bot.edit_message_text(chat_id=msg.chat.id, text= f"{flat_timer_msg_list[workout][global_dict['counter']]}\n\n{timer_piece}",
                                          message_id=msg_timer.message_id, parse_mode=pm["mdV2"])
                except:
                    pass

                secs_before = time.perf_counter()
                secs_taken = secs_before - secs_after

                time.sleep(max(refresh_rate-secs_taken, 0.01))  # refresh_rate - secs_taken serve epr dormire precisamente 1 secondo, max() serve eper evitare numeri negativi
                secs_after = time.perf_counter()

            # se invece refreshed_num_ora + refresh_rate è maggiore fermiamoci per il tempo rimanente e poi passiamo a FINE TIMER
            else:
                refreshed_num_ora = precise_numero_ora()
                time.sleep(max(numOra_fineTimer-refreshed_num_ora, 0.01))  # 0.1 è per sicurezza in modo da non fare mai sleep di un tempo negativo


        # FINE TIMER
        else:
            # COUNTER REMOVING FROM LIST OF COUNTERS TO GO TROUGH
            completed_counters[workout].append(counter)
            levers_dict["timer"] = False

            #eliminazione messaggio timer
            bot_tryDelete(a_uid, msg_timer.message_id)

            for ese in range(len(counter_dict[workout])):
                if counter + 1 == counter_dict[workout][ese]:
                    switch_list[workout][ese] = ""

            last_set = switch_list[workout] == switch_list_quotes[workout]

            # messaggio e notifica
            if levers_dict["skip"] == False or last_set:
                noti = flat_timer_msg_list[workout][global_dict['counter']].replace("/", "").replace("*", "").replace("_", "").replace("`", "")
                client.send_message(f"🔴 Fine timer\n{noti}", title="Timer")

                msg_fineTimer = bot_trySend(f"{flat_timer_msg_list[workout][global_dict['counter']]}\n\n🔴 *Timer finito*", keyboard_done)
                global_dict["msg_fine-timer"] = msg_fineTimer.message_id


            if levers_dict["skip"] == True:
                if switch_list[workout] != switch_list_quotes[workout]:  # solo se l'allenamento non è finito, altrimenti da errore
                    done(msg, workout)
                levers_dict["skip"] = False

            # SE L'ALLENAMENTO è FINITO
            if last_set:
                time.sleep(end_sleepTime)
                bot_tryDelete(a_uid, msg_fineTimer.message_id)

                fine_allenamento(workout, a_uid)
                saveSend_TP(workout)

            break  # interrompe il while loop a "fine timer"


# THREAD MESSAGGI RICEVUTI

def thread_telegram():
    def listener(messages):
        for msg in messages:
            # print(f"{msg =}")
            # print(f"{usermode[a_uid] =}")
            # print(f"{sequences_usermode[a_uid] =}")
            uid = msg.chat.id

            if uid == a_uid:
                msg_id = msg.message_id
                workout = usermode[uid]

                if msg.text == "/test":
                    print("test")
                    ic(msg_id)
                    ic(msg)
                    global_dict["inizioAll_HH:MM:SS"] = "13:45:44"
                    # bot.send_message(uid, text=f"{msg_scheda['addominali']}\n🧭 Fine prevista: {secs_to_time(sum(avg_adj_prev_tempo[workout), False)}",
                    #                  parse_mode=None)

                    keyboard_workous = types.ReplyKeyboardMarkup()
                    keyboard_workous.row(types.KeyboardButton("Addominali"))
                    keyboard_workous.row(types.KeyboardButton("Petto Bi"), types.KeyboardButton("Schiena"))
                    # buttons = [types.InlineKeyboardButton("ciao")]/
                    # keyboard = types.InlineKeyboardMarkup(row_width=1)

                    a = bot.send_message(uid, "OK", reply_markup=keyboard_workous)

                    model_name = "hpo_model"
                    # zipped_model_name = f"{db['models'][model_name]['name']} V{db['models'][model_name]['version']}"
                    zipped_model_name = f"model zip"
                    shutil.make_archive(zipped_model_name, 'zip', f"msg labelling/{model_name}")
                    file = open(f"msg labelling/training_data.json", "rb")
                    bot.send_document(uid, file)

            # """COMANDI GENERALI"""

                # /START
                # TODO, DIFFERENZA TRA INIZIO ALLENAMENTO E RESTART REPL
                elif msg.text == all_commands_dict["/start"]:
                    # se lever_restart è falsa, la mettiamo come vera e attiviamo il bot
                    if levers_dict["restart"] == False:

                        # PARTE DI CREAZIONE MESSAGGIO DI START

                        data, orario = ora_EU(isoformat=True).split("T")
                        oggi_mezzanotte_iso = dt.datetime.fromisoformat(data + "T23:59:59")  # mettiamo la data di oggi ma a mezzanotte in modo di ottenere delle differenze di
                            # giorni precise. es: data_allenamento: 15/2/2022 18:50  il giorno dopo...  data_eu = 16/2/2022 17:30   (data_allenamento - data_eu).days = 0 (!)
                        ultimi_allenamenti_workouts = {}

                        # loop in cui creiamo le keys di ultimi_allenamenti, le key rappresentano [key] giorni fa
                        db_allenamenti = [db["allenamenti"][i] for i in range(len(db["allenamenti"])-1, 1, -1)]
                        ultimi_giorni = None

                        for lista in db_allenamenti:
                            giorni_fa = (oggi_mezzanotte_iso - dt.datetime.fromisoformat(lista[0])).days
                            if giorni_fa == ultimi_giorni or ultimi_giorni == None:
                                ultimi_allenamenti_workouts[giorni_fa] = []  # aggiungiamo la key dei giorni di distanza

                                if len(ultimi_allenamenti_workouts) == 4:  # se siamo arrivati a 4 giorni diversi interrompiamo perchè abbiamo tutto quello che ci serve per gli ultimi
                                        # allenamenti
                                    ultimi_giorni = giorni_fa
                            else:
                                break

                        # loop in cui appendiamo i nomi nel dizionario usando la key di [key] giorni fa
                        for lista_ in db_allenamenti:
                            giorni_fa = (oggi_mezzanotte_iso - dt.datetime.fromisoformat(lista_[0])).days
                            if giorni_fa <= list(ultimi_allenamenti_workouts.keys())[-1]:  # se giorni fa è minore del giorno più lontano registrato in ultimi_allenamenti_workouts
                                ultimi_allenamenti_workouts[giorni_fa].append(nomi_in_scheda[lista_[1]])
                            else:
                                break

                        msg_start = f'*Inizio scheda attuale:* ` {int(global_dict["data_scheda_attuale"][-2:])} {m_to_mesi[global_dict["data_scheda_attuale"][-5:-3]]}`\n\n'
                        msg_start += "*Ultimi allenamenti:*\n    "

                        ultimi_allenamenti_strings = []
                        for giorni_fa, nomi_w in ultimi_allenamenti_workouts.items():
                            string = ""
                            for i, nome in enumerate(nomi_w):
                                if len(nomi_w) - (i+1) == 0:  # se non seguono altri nomi per questo giorno
                                    string += f"*{nome}*\n"
                                else:
                                    string += f"*{nome}*, "
                            string += f"        `{weekday_nday_month_year(oggi_mezzanotte_iso - dt.timedelta(days=giorni_fa), weekday=True)}`, _{giorni_fa} giorni fa_\n    "
                            ultimi_allenamenti_strings.append(string)
                        ultimi_allenamenti_strings.reverse()  # reversiamo così che gli ultimi giorni saranno primi

                        msg_start += "".join(stringa for stringa in ultimi_allenamenti_strings)
                        msg_start = msg_start.replace(" _0 giorni fa", " _Oggi").replace(" _1 giorni fa", " _Ieri")


                        raw_buttons = dict(texts=[val for _, val, in esercizi_title_dict.items()],  # tutti gli esercizi
                                           callbacks=[f"WORK{key}"  for key, _, in esercizi_title_dict.items()])
                        global_dict["callback_antiSpam"] = True
                        buttons = inline_buttons_creator(raw_buttons, 2)

                        msg_ = bot.send_message(msg.chat.id, text=msg_start, reply_markup=buttons, parse_mode=pm["md"])
                        global_dict["msg_sceltaAll"] = msg_.message_id

                        levers_dict["restart"] = True
                        global_dict["counter"] = 0
                        global_dict["start_message_id"] = msg_id
                        global_dict["callback_antiSpam"] = False

                    # se lever_restart è vera, la mettiamo come falsa e riavviamo il bot, quindi torna falsa
                    else:
                        levers_dict["restart"] = False

                        # if not actual_environment:
                        #     with open("temporanei/cloned_db.json", "w") as file:
                        #         json.dump(db, file, indent=4)

                        os.execv(sys.executable, ["python"] + sys.argv)


                # /COMANDI
                elif msg.text == all_commands_dict["/comandi"]:

                    msg = "<b>Comandi generali</b>"
                    for comando, desc in dict_comandi_gen.items():
                        msg += md_v2_replace(f"\n`{comando}` - {desc}", exceptions=["`"])
                    msg += "\n<b>Comandi durante allenamento</b>"
                    for comando, desc in dict_comandi_dur.items():
                        msg += md_v2_replace(f"\n`{comando}` - {desc}", exceptions=["`"])

                    bot.send_message(uid, msg, parse_mode=pm["HTML"])

                # /SINTASSI
                elif msg.text == all_commands_dict["/sintassi"]:
                    msg_sintassi = """
                    *isomix* - (8rep, tenere 5s) tre volte
                    *;* - separatore dall'ultima rep
                    *,* - separatore normale di rep
                    *5s* - 5r e scalo
                    *t 5\"* - tieni 5s
                    */* - separatore di superset
                    *h* - orizzontale
                    *p:5* - panca con inclinazione 5
                    *5|7* - massimo di 7 rep, minimo di 5
                    *10|8|6|4* - prima serie: 10 rep, peso x; seconda serie: 8 rep, peso x+y ... 
                    *V* - barra a forma di V
                    *5rp* - alla fine delle ripetizioni pausa 10", poi massimo delle rep, tutto questo 5 volte (prima rep non inclusa)
                    *mp* - macchinario bilancere
                    *h* - presa a martello (hammer)
                    """
                    bot.send_message(uid, msg_sintassi, parse_mode=pm["md"])


                # /EXCEL
                elif msg.text == all_commands_dict["/excel"]:
                    # todo per questo serve conversione da db a excel (con xlswriter)
                    # excelscheda = open(scheda_attuale_path, "rb")
                    # msg_ = bot.send_document(chat_id=msg.chat.id, document= excelscheda)
                    # db["preserved_msgIds"].append(msg_.message_id)
                    bot.send_message(uid, "per ora la conversione db -> excel non è supportata, usare /archivio per visualizzare la scheda")


                # /DATABASE
                elif msg.text == all_commands_dict["/database"]:
                    send_db_json()
                    bot.send_message(uid, "per informazioni su come dovrebbe essere il database usare /info")

                # /INFO
                elif msg.text == all_commands_dict["/info"]:
                    info = """
                    *⚠️ Informazioni utili ⚠️*
                    Database
                    •  `data_prossima_stat` dovrebbe essere di domenica e a 4 settimane dall'ultima statistica di 4 settimane
                    •  `calendario` dovrebbe essere il mese in cui siamo in questo momento
                    •  `data_prossima_stat_m` dovrebbe essere l'ultimo giorno di questo mese
                    •  `Cycle_dailyLever` dovrebbe essere True se l'orario è prima delle 22:00, altrimenti False
                    •  `Cycle_4settLever` e `Cycle_monthLever` devono essere True se non è stato inviato un grafico entro 2 ore fa
                    
                    Scrittura peso
                    •  usare "k" per indicare i kg, tutto il resto sarà assolutamente sbagliato
                       es:  NO: 36kg  si: 36k
                    •  lasciare sempre un " " dopo un peso quando è seguito da altro.
                       es:  NO: 36k,35k,  SI: 36k, 35k
                    •  usare sempre k per indicare i kg, visto che se non lo mettiamo si parla di "dischi in più dei macchinari"
                       es:  NO: 36        SI: 36k
                    •  usare + per indicare che usiamo anche dei "dischi in più dei macchinari" (il + non è necessario, potremmo anche scrivere [nuumero_kg]k [numero_dischi_extra] ma non è
                       molto comprensibile
                       es:  36k + 2  (36k 2 funziona lo stesso)
                    •  mettere le parentesi per indicare le diminuizioni di peso nella stessa serie
                       es:  reps: 8 + 8 + 8  peso: SI:  36k (- 1) oppure 36k (-1)  NO: 36k - 1
                    •  mettere sempre i kg usati in un solo numero:
                       es:  NO: 20k 15k (che vorrebbe dire 35kg in totale)  SI: 35k
                
                    Info peso
                    •  tutte le cose che non sono numeri verranno completamente ignorate, quindi se vengono separate correttamente con " " dai kg, non creeranno problemi
                    •  tutti i pezzi contenenti ( o ) vengono ignorati perchè non servono
                    """
                    bot_trySend(info)


                # 1 /NUOVASCHEDA
                elif msg.text == all_commands_dict["/nuovascheda"]:
                    cambiaScheda_button = dict(texts=["Cambia scheda"], callbacks=["NUOV"])
                    button = inline_buttons_creator(cambiaScheda_button, 1)
                    bot.send_message(msg.chat.id, text="Sei sicuro di voler cambiare scheda? Usare questo comando solo se si cambia la scheda, se lo hai già fatto per questa "
                                                       "scheda non rifarlo, ti basterà selezionare gli esercizi da compilare con /start",
                                     reply_markup=button)


                # 1 /ARCHIVIO
                elif msg.text == all_commands_dict["/archivio"]:
                    date_schede = list(startup_db["schede"].keys())
                    date_schede.sort()
                    global_dict["schede_list"] = date_schede
                    raw_buttons = {"texts": [], "callbacks": []}

                    for scheda in date_schede:
                        anno_inizio, sett, data_inizio, data_fine, index, ultima_scheda = schedaFname_to_title(scheda, date_schede)

                        if ultima_scheda == False:
                            raw_buttons["texts"].append(f"{anno_inizio}:  {weekday_nday_month_year(data_inizio)} - {weekday_nday_month_year(data_fine)}, {sett} sett")
                        else:
                            raw_buttons["texts"].append(f"Attuale: {weekday_nday_month_year(data_inizio)}, {sett} sett")

                        raw_buttons["callbacks"].append(f"ARCH{index}")

                    buttons = inline_buttons_creator(raw_buttons, row_width=1)
                    global_dict["callback_antiSpam"] = False
                    bot.send_message(uid, "Seleziona quale scheda visualizzare", reply_markup=buttons)


                # /TEMPI
                elif msg.text == all_commands_dict["/tempi"]:
                    global_dict["inizioAll_HH:MM:SS"] = ora_EU(soloOre=True)
                    msg = f"""*MEDIA TEMPO ULTIMI 3 WORKOUT:*\n"""
                    for workout in active_workouts:
                        msg += f"*{nomi_in_scheda[workout]}*: {secs_to_time(sum(avg_adj_prev_tempo[workout]), False)}, {secs_to_hours(sum(avg_adj_prev_tempo[workout]))}"

                    bot.send_message(chat_id=uid, text=msg, parse_mode=pm["md"])


                # 1 /SCORE
                elif msg.text == all_commands_dict["/score"]:
                    bot.send_message(msg.chat.id, text="Seleziona l\'esercizio che vuoi segnare", reply_markup=keyboard_workouts)
                    sequences_usermode[uid] = "score"

                # 2
                elif sequences_usermode[uid] == "score":
                    if msg.text in nomi_workout:
                        fine_allenamento(start_dict[msg.text], uid, True)

                    else:
                        bot.send_message(msg.chat.id, text=f"Seleziona un esercizio valido", reply_markup=keyboard_workouts)


                # /MSGDEBUG
                elif msg.text == all_commands_dict["/msgdebug"]:
                    global pm_toggle
                    if pm_toggle == False:
                        pm["mdV2"] = None
                        pm["HTML"] = None
                        pm["md"] = None

                        pm_toggle = True
                        bot.send_message(uid, "Parse modes rimosse, usare di nuovo /msgdebug per riattivare le parse modes")

                    else:
                        pm["mdV2"] = "MarkdownV2"
                        pm["HTML"] = "HTML"
                        pm["md"] = "Markdown"
                        bot.send_message(uid, "Parse modes riaggiunte")


                # /EXIT
                elif msg.text == all_commands_dict["/exit"]:
                    bot_tryDelete(uid, msg_id)
                    bot_tryDelete(uid, msg_id+1)

                    sequences_usermode[uid] = "none"

                # /FORCE TRAINING
                elif msg.text == "/forcetraining":
                    start_training_procedure()

                # 1 /PESO
                elif msg.text == all_commands_dict["/peso"]:
                    bot.send_message(msg.chat.id, text=f"Inviare il numero di mesi precedenti da cui prendere i dati")
                    sequences_usermode[uid] = "Grafico_peso"

                # 2     /PESO
                elif sequences_usermode[uid] == "Grafico_peso":
                    valid = False
                    try:
                        num_months = int(msg.text)
                        global_dict["num_months"] = num_months

                        valid = True
                    except:
                        bot.send_message(msg.chat.id, text=f"Inviare un numero valido")

                    if valid:
                        levers_dict["grafico_peso"] = True


            # DURANTE ESERCIZIO

                elif usermode[uid] in lista_es_simple:

                    # /DONE
                    if msg.text == all_commands_dict["/done"]:
                        if not levers_dict["timer"]:  # se il timer non sta già andando
                            done(msg, workout)
                        else:
                            bot.delete_message(uid, msg_id)


                    # /NOTIME
                    elif msg.text == all_commands_dict["/notime"]:
                        levers_dict["notime"] = True
                        bot.send_message(uid, "Il tempo che verrà impiegato per finire questo allenamento non verrà salvato")


                    # SKIP /SK
                    elif msg.text == all_commands_dict["/sk"]:
                        levers_dict["skip"] = True


                        bot.delete_message(chat_id= uid,message_id=msg_id)

                    # /BACK
                    elif msg.text == all_commands_dict["/back"]:
                        global_dict["counter"] -= 1
                        bot.delete_message(chat_id= uid,message_id=msg_id)


                    # 1 SWITCH /SW
                    elif msg.text == all_commands_dict["/sw"]:
                        bot.delete_message(chat_id=uid, message_id = msg_id)

                        keyb_switch = types.ReplyKeyboardMarkup(one_time_keyboard=True)
                        for ese in range(len(switch_list[workout])):
                            keyb_switch.add(switch_list[workout][ese])
                        msg_switch = bot.send_message(msg.chat.id, text=f"Scegli a quale esercizio andare", reply_markup=keyb_switch)

                        global_dict["switch_msg"] = msg_switch.message_id
                        sequences_usermode[uid] = "switch"

                    # 2     SECONDO PASSO SWITCH
                    elif sequences_usermode[uid] == "switch":
                        index_sw = switch_list_full[workout].index(msg.text)

                        contatore_cdt = 0
                        counter_dict_timer_full[workout].clear()

                        for ese in range(len(serie_in_ese_list[workout])):
                            counter_dict_timer_full[workout].append([])
                            for serie in range(len(serie_in_ese_list[workout][ese])):
                                counter_dict_timer_full[workout][ese].append(contatore_cdt)
                                contatore_cdt += 1


                        cdt_index_pop = counter_dict_timer_full[workout].pop(index_sw)
                        count_serie[workout].remove(cdt_index_pop)
                        count_serie[workout].insert(0, cdt_index_pop)

                        # portiamo il counter al numero compatibile all'esercizio
                        list_sw = []
                        if index_sw > 0:
                            for i in range(index_sw):
                                list_sw.append(diz_all[workout][i][3])
                                global_dict["counter"] = sum(list_sw)
                        else:
                            global_dict["counter"] = 0

                        sequences_usermode[uid] = "none"

                        bot_tryDelete(uid, msg_id)
                        bot_tryDelete(uid, global_dict["switch_msg"])


                    # MODIFICHE ALLA SCHEDA

                    # 1, /comando
                    elif msg.text in ["/p", "/ese", "/n", "/rep"]:
                        bot_tryDelete(uid, msg_id)

                        msg_ = bot.send_message(uid, text=modifiche_dict[msg.text]["msg0"], reply_markup=modifiche_dict[msg.text]["kb0"][workout], parse_mode=pm["md"])
                        global_dict["msg_mod"] = msg_.message_id
                        sequences_usermode[uid] = msg.text + "1"  # /ese1

                    # 2, alla selezione del nuovo esercizio
                    elif sequences_usermode[uid] in ["/p1", "/ese1", "/n1", "/rep1"]:
                        bot_tryDelete(uid, msg_id)
                        bot_tryDelete(uid, global_dict["msg_mod"])

                        tipo_modifica = sequences_usermode[uid][:-1]  # /ese
                        iteration_list = modifiche_dict[tipo_modifica]["iter_list"]

                        for ese in range(len(iteration_list[workout])):
                            if msg.text == iteration_list[workout][ese]:
                                nome_ese = switch_list_full[workout][ese]
                                msg_ese = f"Esercizio selezionato: *{nome_ese}*"

                                if sequences_usermode[uid] == "/p1":
                                    msg_ = bot.send_message(msg.chat.id,
                                                            text=f"{msg_ese}\n`{diz_all[workout][ese][4].replace('㎏', 'k')}`, scrivere il nuovo peso. \n"
                                                                 f"Scrivere il peso seguendo le regole sulla scrittura del peso che si trovano su /info",
                                                            parse_mode=pm['md'])
                                elif sequences_usermode[uid] == "/ese1":
                                    msg_ = bot.send_message(msg.chat.id, text=f"Esercizio selezionato: `{nome_ese}`, scrivere il nuovo nome",
                                                            parse_mode=pm["md"])
                                elif sequences_usermode[uid] == "/rep1":
                                    msg_ = bot.send_message(msg.chat.id, text=f"{msg_ese}, scrivere le nuove reps",
                                                            parse_mode=pm["md"])
                                elif sequences_usermode[uid] == "/n1":
                                    msg_ = bot.send_message(msg.chat.id, text=f"{msg_ese}, aggiungere le note",
                                                            parse_mode=pm["md"])

                                global_dict["msg_mod"] = msg_.message_id
                                global_dict["idx_es_cambiato"] = ese
                                sequences_usermode[uid] = tipo_modifica + "2"  # /ese2

                                break

                    # 3, alla scrittura del nuovo valore
                    elif sequences_usermode[uid] in ["/p2", "/ese2", "/n2", "/rep2"]:
                        bot_tryDelete(uid, msg_id)
                        bot_tryDelete(uid, global_dict["msg_mod"])

                        tipo_modifica = sequences_usermode[uid][:-1]  # /ese

                        vecchio_peso = copy.deepcopy(diz_all[workout][global_dict['idx_es_cambiato']][modifiche_dict[tipo_modifica]["db_idx"]])
                        vecchio_nome_ese = diz_all[workout][global_dict['idx_es_cambiato']][modifiche_dict[tipo_modifica]["db_idx"]]
                        vecchia_rep = diz_all[workout][global_dict['idx_es_cambiato']][modifiche_dict[tipo_modifica]["db_idx"]]
                        vecchie_note = diz_all[workout][global_dict['idx_es_cambiato']][modifiche_dict[tipo_modifica]["db_idx"]]

                        # scrittura in db
                        apply_modification(msg.text, workout, global_dict["idx_es_cambiato"], modifiche_dict[tipo_modifica]["db_idx"])

                        nome_ese = diz_all[workout][global_dict['idx_es_cambiato']][1]
                        peso_ese = diz_all[workout][global_dict['idx_es_cambiato']][4]
                        note = diz_all[workout][global_dict['idx_es_cambiato']][5]
                        rep = diz_all[workout][global_dict['idx_es_cambiato']][2]

                        # /p
                        if sequences_usermode[uid] == "/p2":

                            # aumento peso per grafici
                            # db["aumenti_peso"].append([ora_EU(isoformat=True),
                            #                            vecchio_peso,
                            #                            msg.text,
                            #                            nome_ese])
                            db_insert("aumenti_peso", [ora_EU(isoformat=True),
                                                       vecchio_peso,
                                                       msg.text,
                                                       nome_ese],
                                              mode="append")

                            # messaggio aumento / diminuizione peso
                            if stringaPeso_to_num(vecchio_peso) < stringaPeso_to_num(msg.text):
                                reply_msg = f"💹 _{nome_ese}_: {vecchio_peso} *→* {peso_ese}"
                                db_insert(["aumenti_msgScheda", workout], {"w_count": db["workout_count"][workout],
                                                                           "ese_idx": global_dict["idx_es_cambiato"],
                                                                           "peso": vecchio_peso},
                                          mode="append")
                                # db["aumenti_msgScheda"][workout].append({"w_count": db["workout_count"][workout],
                                #                                          "ese_idx": global_dict["idx_es_cambiato"],
                                #                                          "peso": vecchio_peso})

                            else:
                                reply_msg = f"🈹 _{nome_ese}_: {vecchio_peso} *→* {peso_ese}"

                        # /ese
                        elif sequences_usermode[uid] == "ese2":
                            reply_msg = f"_{vecchio_nome_ese}_ *→* _{nome_ese}_"

                        elif sequences_usermode[uid] == "/rep2":
                            reply_msg = f"_{vecchia_rep}_ *→* _{rep}_"

                        # /n
                        elif sequences_usermode[uid] == "/n2":
                            # se c'erano già delle note per questo esercizio
                            if vecchie_note != None:
                                reply_msg = f"Note di _{nome_ese}_ cambiate da _{vecchie_note}_ a _{note}_"
                            else:
                                reply_msg = f"Note di _{nome_ese}_ aggiunte: _{note}_"

                        msg_ = bot.send_message(uid, text=reply_msg, parse_mode=pm["md"], reply_markup=keyboard_done)
                        preserve_msg(msg_)

                        # nuovi dati per il modello
                        db_insert("new_training_data", [msg.text, modifiche_dict[tipo_modifica]["label"]], mode="append")
                        # db["new_training_data"].append([msg.text, modifiche_dict[tipo_modifica]["label"]])

                        sequences_usermode[uid] = "none"

                    # CREAZIONE SCHEDA

                    # /FINE
                    elif msg.text == all_commands_dict["/fine"]:
                        db["schede"][global_dict["data_scheda_attuale"]][workout] = [ese for ese in diz_all[workout] if ese[5] != False]  # mettiamo nel db tutti gli esercizi di
                            # diz_all che non hanno le note come "False", cioè che indicano che l'esercizio è vuoto e struttura base

                        # nuovi dati di training
                        for ese in db["schede"][global_dict["data_scheda_attuale"]][workout]:
                            for i in [1, 2, 4, 5]: # solo gli index a cui viene applicato il ML
                                if i == 1 and "° Esercizio" not in ese[i]:
                                    # db["new_training_data"].append([ese[i], "nomi"])
                                    db_insert("new_training_data", [ese[i], "nomi"], mode="append")
                                elif i == 2 and ese[i] != "/":
                                    # db["new_training_data"].append([ese[i], "reps"])
                                    db_insert("new_training_data", [ese[i], "reps"], mode="append")
                                elif i == 4 and ese[i] != "/":
                                    # db["new_training_data"].append([ese[i], "peso"])
                                    db_insert("new_training_data", [ese[i], "peso"], mode="append")
                                elif i == 5 and ese[i] != None:
                                    # db["new_training_data"].append([ese[i], "note"])
                                    db_insert("new_training_data", [ese[i], "note"], mode="append")

                        fine_allenamento(usermode[uid], uid)
                        saveSend_TP(workout)

                    # COMPILAZIONE SCHEDA
                    else:
                        if global_dict["creazione_scheda"] and msg.text[0] != "/":
                            i_ese = global_dict["i_ese"]

                            # qualsiasi modifica facciamo a questo step trasforma l'esercizio in un esercizio reale, facendo passare note da False a None
                            diz_all[workout][i_ese][5] = None
                            bot.delete_message(uid, msg.id)

                            # SERIE
                            if msg.text in str_1_6_list:  # format corretto: N
                                cambioScheda_compile(msg.text, workout, i_ese, 3, uid)

                            # PAUSA
                            elif msg.text[-1] == "\"" and is_digit(msg.text[:-1]):  # format corretto: NN"
                                cambioScheda_compile(msg.text, workout, i_ese, 0, uid)

                            # CASO "N REPS"
                            elif msg.text.endswith("reps") and len(msg.text) < 7:
                                cambioScheda_compile(msg.text, workout, i_ese, 2, uid, forced_reps=True)

                            # NOME, REPS, PESO, NOTE
                            else:
                                # pred
                                model_input = tf.expand_dims(tf.constant(msg.text), 0)
                                pred = global_dict["msgLabelling_model"](model_input)
                                pred_idx = int(tf.argmax(pred, 1))
                                type = int_to_type[pred_idx]  # Nome, Reps, ...

                                # testo da aggiungere al messaggio
                                probs_dict = {key: round(float(pred[0][i]), 2) for key, i in zip(["E", "R", "P", "N"], range(4))}
                                s_probs_dict = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
                                probs_text = [f"{key} {prob:.0%}" for key, prob in s_probs_dict.items()]
                                probs_text = "  ".join(probs_text)

                                # bottoni per cambiare tipo
                                texts_list = [letter_to_type[key] for key in s_probs_dict][1:] + ["✖️ Undo"]
                                change_buttons_raw = dict(texts=texts_list, callbacks=[f"CAMB{pred_idx}{i}" for i in texts_list])  # il callback è composto dal prev_idx del tipo
                                    # previsto e dalla stringa del tipo selezionato per cambiare
                                change_buttons = inline_buttons_creator(change_buttons_raw, 4)
                                global_dict["cS_callback_antiSpam"] = False
                                creazioneScheda_undo["cambioScheda_msg"] = msg.text

                                # rimozione buttons a scorso messaggio di predizione tipo
                                if creazioneScheda_undo["cambioScheda_msgId"] != None:  # solo se sono già stati mandati di compilazione
                                    try:  # try block perchè potrebbe essere un undo e quindi il messaggi oè stato elimiato
                                        bot.edit_message_reply_markup(uid, creazioneScheda_undo["cambioScheda_msgId"], reply_markup=None)
                                    except:
                                        pass

                                msg_ = cambioScheda_compile(msg.text, workout, i_ese, type_to_dizAll_idx[type], uid, change_buttons=change_buttons, bot_probs=probs_text)
                                creazioneScheda_undo["cambioScheda_msgId"] = msg_.message_id


                        else:
                            bot.send_message(msg.chat.id, text=f"Comando sconosciuto")


                # non durante allenamento ma usa comando dove è richiesto essere in allenamento
                elif msg.text in dict_comandi_dur:
                    bot.send_message(msg.chat.id, text=f"Per usare {msg.text} devi prima selezionare un allenamento con /start")


                # SCORE PESO, FUORI DAL TRAINING
                elif msg.text[2] in [".", ","] and len(msg.text) == 5:
                    for i in [0,1,3,4]:  # index dei numeri in msg.text
                        if msg.text[i] in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
                            real_num = True
                        else:
                            real_num = False
                            break

                    if real_num:
                        peso_float = float(msg.text.replace(",", "."))
                        data = ora_EU()
                        # db["peso_bilancia"].append([data.isoformat(), peso_float])
                        db_insert("peso_bilancia", [data.isoformat(), peso_float], mode="append")

                        msg_ = bot.send_message(msg.chat.id, text=f"✅ Peso segnato con successo ({peso_float}㎏)")
                        preserve_msg(msg_)
                        # db["preserved_msgIds"].append(msg_.message_id)
                    else:
                        bot.send_message(msg.chat.id, text="Il format da seguire per registrare il peso è: <code>NN.NN</code> o <code>NN,NN</code>", parse_mode=pm["HTML"])


                # ERRORE
                else:
                    bot.send_message(msg.chat.id, text="Comando sconosciuto")


    ##################


    # CALLBACK HANDLER

    @bot.callback_query_handler(func=lambda call: True)
    def callback(call):
        uid = call.message.chat.id
        msg_id = call.message.id
        code = call.data[0:4]
        after_code = call.data[4::]


        # INIZIO WORKOUT e NUOVA SCHEDA

        if code == "WORK":
            global TP_data
            if global_dict["callback_antiSpam"] == False:
                global_dict["callback_antiSpam"] = True  # il motivo di questa lever è impedire che quando la richiesta viene spammata per un bug (si verifica quando mandiamo
                    # la richiesta e usciamo da telegram, a quel punto continuano a venire mandate richieste) si azioni la procedura di start

                bot.edit_message_reply_markup(uid, global_dict["msg_sceltaAll"], reply_markup=None)

                global_dict["inizioAll_HH:MM:SS"] = ora_EU(soloOre=True)
                global_dict["inizioAll_secs"] = time.perf_counter()

                workout = after_code
                usermode[a_uid] = workout

                # ORDINARIO
                if db["schede"][global_dict["data_scheda_attuale"]][workout] != strutturaBase_dizAll:  # nel caso in cui il workout è già stato compilato
                    n_ese = len(diz_all[workout])
                    msg_scheda, msg_schedaPrev = crea_messaggioScheda(workout, send=True)
                    global_dict["msg_scheda"], global_dict["msg_schedaPrev"] = msg_scheda.message_id, msg_schedaPrev.message_id  # il messaggio della previsione del tempo non viene
                        # salvato (_) e in questo modo non viene eliminato e rimane la prima previsione del bot che alla fine si può confrontare con il risultato finale

                    TP_data = dict(timedate=ora_EU(isoformat=True), workout=after_code, n_workout=db["workout_count"][workout]+1,  # info
                                   S_pause=[diz_all[workout][i][0] for i in range(n_ese)],
                                   S_nomi =[diz_all[workout][i][1] for i in range(n_ese)],
                                   S_reps =[diz_all[workout][i][2] for i in range(n_ese)],
                                   S_serie=[diz_all[workout][i][3] for i in range(n_ese)],
                                   ultimi_allenamenti=startup_db["w_sec_taken"][workout],
                                   current=None)

                # NUOVA SCHEDA
                else:
                    global_dict["creazione_scheda"] = True  # serve epr ottenere accesso all'area in fondo a thread telegram durante esercizio
                    global_dict["i_ese"] = 0

                    # modello DL
                    msg_ = bot.send_message(uid, "Importazione dell'intelligenza artificiale...")
                    msg_clessidra = bot.send_message(uid, text="⏳")

                    global tf
                    import tensorflow as tf

                    model_name = "hpo_model"
                    global_dict["msgLabelling_model"] = tf.keras.models.load_model(f"msg labelling/{model_name}")
                    plot_name = f"immagini/model_plot.png"
                    # v- NON funziona su replit
                    # tf.keras.utils.plot_model(global_dict["msgLabelling_model"], to_file=plot_name, show_shapes=True, show_layer_activations=True)
                    bot.send_photo(uid, photo=open(plot_name, "rb"), caption=f"🤖 Modello in uso: {db['NNs_names'][model_name]['name']} V{db['NNs_names'][model_name]['version']}")

                    bot.delete_message(uid, msg_.message_id)
                    bot.delete_message(uid, msg_clessidra.message_id)

                    # stats per radar graph
                    db_insert(["workout_count", workout], 0)
                    db_insert(["data_2volte", workout], None)
                    db_insert(["dizAll_2volte", workout], None)
                    # db["workout_count"][workout] = 0
                    # db["data_2volte"][workout] = None
                    # db["dizAll_2volte"][workout] = None

                    # messaggio di aiuto e messaggio scheda
                    bot.send_message(uid, text="Info sulla creazione scheda:\n"
                                               "•  Format per le serie: 'N', (N: num da 1 a 6)\n"
                                               "•  Format per la pausa: 'NN\"' (N: num)\n"
                                               "•  Nome esercizio, reps, peso e note sono gestite dal neural network. nel caso in cui si voglia usare un numero di reps che va in "
                                               "conflitto con le regole per le serie, scrivere \"N reps\" e verrà preso solo il numero N\n"
                                               "•  All'ultima serie dell'ultimo esercizio, dopo aver fatto tutte le modifiche, fare /fine, se non si fa questo ultimo passaggio "
                                               "nessuna modifica viene applicata", reply_markup=keyboard_done)

                    # db["data_cambioscheda"].append(ora_EU(isoformat=True))
                    db_insert("data_cambioscheda", ora_EU(isoformat=True), mode="append")
                    msg_scheda = bot_trySend(f"{messaggio_scheda(workout)}")
                    global_dict["msg_scheda"] = msg_scheda.message_id


        # CONFERMA DI /NUOVASCHEDA
        elif code == "NUOV":
            # creazione struttura nuova scheda nel db
            YYYY_MM_DD = str(ora_EU())[:10]
            db_insert(["schede", YYYY_MM_DD],
                      dict(addominali=strutturaBase_dizAll, pettobi=strutturaBase_dizAll, schiena=strutturaBase_dizAll,
                           spalletri=strutturaBase_dizAll, altro=strutturaBase_dizAll))
            # db["schede"][YYYY_MM_DD] = dict(addominali=strutturaBase_dizAll, pettobi=strutturaBase_dizAll, schiena=strutturaBase_dizAll,
            #                                 spalletri=strutturaBase_dizAll, altro=strutturaBase_dizAll)
            for workout in lista_es_simple:
                # reset prev tempo
                # db["w_sec_taken"][workout] = []
                db_insert(["w_sec_taken", workout], [])

            global_dict["data_scheda_attuale"] = get_dataSchedaAttuale()
            levers_dict["restart"] = True

            # GRAFICO RADAR SCHEDA PRECEDENTE
            levers_dict["radar_graph"] = True

            # db data cambioscheda
            # db["data_cambioscheda"].append(ora_EU(isoformat=True))
            db_insert("data_cambioscheda", ora_EU(isoformat=True), mode="append")

            bot.send_message(uid, text="Scrivere /start per iniziare a compilare i nuovi workout", parse_mode=pm["md"])


        # CAMBIO TIPO IN CREAZIONE SCHEDA e /UNDO
        elif code == "CAMB":
            if global_dict["cS_callback_antiSpam"] == False:
                global_dict["cS_callback_antiSpam"] = True

                # es call.data: CAMB0Reps  da 0 (nome) si vuole passare a reps
                previous_dizAll_idx = type_to_dizAll_idx[int_to_type[int(after_code[0])]]  # idx dell'esercizio che vogliamo cambiare
                new_dizAll_idx = type_to_dizAll_idx[after_code[1:]] if after_code[1:] in ["Nome", "Reps", "Peso", "Note"] else None # idx del nuovo esercizio

                bot_tryDelete(uid, creazioneScheda_undo["cambioScheda_msgId"])
                cambioScheda_compile(creazioneScheda_undo["previous_val"], usermode[uid], global_dict["i_ese"], previous_dizAll_idx, uid,
                                     callback=True, CB_new_dizAll_idx=new_dizAll_idx, CB_new_val=creazioneScheda_undo["cambioScheda_msg"])

        # SCHEDA ARCHIVIO
        elif code == "ARCH":
            if global_dict["callback_antiSpam"] == False:
                index = int(after_code)
                scheda_YYYY_MM_DD = global_dict["schede_list"][index]
                img_path = f"schede/immagini/{scheda_YYYY_MM_DD[:-5]}.jpeg"

                if os.path.isfile(img_path) and index != len(global_dict["schede_list"]) - 1:  # se c'è già l'immagine e non è la scheda attuale
                    loaded_img = open(img_path, "rb")
                    bot.send_document(uid, loaded_img)

                else:
                    msg_clessidra = bot.send_message(uid, text="⏳")

                    loaded_img = schedaIndex_to_image(startup_db["schede"][scheda_YYYY_MM_DD], scheda_YYYY_MM_DD)

                    bot.delete_message(uid, msg_clessidra.message_id)
                    msg_ = bot.send_document(uid, loaded_img)
                    preserve_msg(msg_)
                    # db["preserved_msgIds"].append(msg_.message_id)

                global_dict["callback_antiSpam"] = True  # ha la stessa funzionalità di callback_antiSpam




    #################################


    bot.set_update_listener(listener)


    while True:
        # bot.polling(none_stop=True)  # per debug: uncommentare questa linea
        try:
            bot.polling(none_stop=True)
        except Exception as exc:
            traceback_message(exc)


t = threading.Thread(target=thread_telegram)
t.start()

addTo_botStarted(f"Functions and packages loaded, messages handling ready")

#######################


# CYCLE (MAIN THREAD)

while True:
    cycle_seconds = 600
    for _ in range(cycle_seconds):
        if levers_dict["grafico_peso"]:
            try:
                fname, msg = Grafico_peso(db["peso_bilancia"], global_dict["num_months"])
                img = open(fname, "rb")
                msg_ = bot.send_photo(a_uid, img, caption=msg, parse_mode=pm["HTML"])
                preserve_msg(msg_)
                sequences_usermode[a_uid] = "none"

            except Exception as exc:
                bot.send_message(a_uid, f"C'è stato un errore ({exc})")
                split_messaggio(a_uid, f"TRACEBACK:\n\n{traceback.format_exc()}")

            levers_dict["grafico_peso"] = False

        elif levers_dict["radar_graph"]:
            try:
                img_grafico_WS = Grafico_radarScheda(db["dizAll_2volte"], diz_all, db["data_2volte"], db["workout_count"])
                bot.send_photo(a_uid, img_grafico_WS)

            except Exception as exc:
                bot.send_message(a_uid, text="Il radar graph ha fallito:", parse_mode=pm["md"])
                traceback_message(exc)

            levers_dict["radar_graph"] = False

        time.sleep(1)

    # time.sleep(5)
    # print("time sleep finished")
    n_ora = numero_ora()
    sec_22 = 60*60*22

    # # se sono passate le 22:00
    if n_ora > sec_22 and db["Cycle_processLever"]:
    # if 1 == 1:
        data_oggi = ora_EU()
        data_oggi_iso = ora_EU(isoformat=True)

        if db["Cycle_dailyLever"] == True:
            # RIMOZIONE MESSAGGI
            clear_msg = bot.send_message(a_uid, "Pulizia dei messaggi inutili in corso...")
            l_msgId = db["latest_msgId"]

            if actual_environment:  # solo se è da replit e non da pycharm, in questo modo non vengono eliminati tutti i messaggi (visto che il db non è quasi mai aggiornato)
                for msg_id in range(l_msgId + 1, clear_msg.message_id + 1):
                    bot_tryDelete(a_uid, msg_id)
                pass
                # db["latest_msgId"] = clear_msg.message_id
                db_insert("latest_msgId", clear_msg.message_id)


            # TRAINING MODELLO
            if len(db["new_training_data"]) > 50:
                start_training_procedure()

            # db["Cycle_dailyLever"] = False
            db_insert("Cycle_dailyLever", False)


        # GRAFICO 4 SETTIMANE

        data_prossima_stat = dt.datetime.fromisoformat(db["data_prossima_stat"])
        # se non ci saranno imprevisti sarà la data in cui il programma crea il grafico delle settimane

        # se sono passati 28 giorni dall'ultima volta che ci sono state le statistiche, oppure: se la data prossima stat nel database è prima di quella di oggi
        # sarebbe oggi ma per sicurezza mettiamo la data che dovrebbe essere per evitare che il grafico venga sballato in caso il processo non venga eseguito nel giorno giusto
        if data_prossima_stat < data_oggi and db["Cycle_4settLever"] == True:
        # if 1 == 1:

            try:
                bot.send_message(a_uid, "GRAFICO 4 SETTIMANE:")

                msg, img, sum_y_allenamenti, sum_y_addominali, sum_y_aumenti_peso = Grafico_4sett(db, data_prossima_stat, somma_numeri_in_stringa, m_to_mesi)
                msg_ = bot.send_photo(a_uid, img, caption=msg, parse_mode=pm["HTML"])
                # db["preserved_msgIds"].append(msg_.message_id)
                preserve_msg(msg_)

                # DB
                # settiamo la data della prossima data nel databse
                # db["data_prossima_stat"] = (data_prossima_stat + dt.timedelta(days=28)).isoformat()
                db_insert("data_prossima_stat", (data_prossima_stat + dt.timedelta(days=28)).isoformat())

                # salviamo nel database i dati per il prossimo grafico mensile
                db_insert(["ultima_stat", "y_allenamenti"], sum_y_allenamenti)
                db_insert(["ultima_stat", "y_addominali"], sum_y_allenamenti)
                db_insert(["ultima_stat", "y_aumenti_peso"], sum_y_aumenti_peso)
                # db["ultima_stat"]["y_allenamenti"] = sum_y_allenamenti
                # db["ultima_stat"]["y_addominali"] = sum_y_addominali
                # db["ultima_stat"]["y_aumenti_peso"] = sum_y_aumenti_peso

                db_insert("Cycle_4settLever", False)
                # db["Cycle_4settLever"] = False

            except Exception as exc:
                msg_ = bot.send_message(a_uid, f"C'è stato un errore nel grafico delle 4 settimane")
                preserve_msg(msg_)
                # db["preserved_msgIds"].append(msg_.message_id)
                msg_ = traceback_message(exc)
                preserve_msg(msg_)
                # db["preserved_msgIds"].append(msg_.message_id)


        # ITERAZIONE MENSILE

        # se i giorni tra oggi e la data in cui ci sarebbe il prossimo grafico sono meno di 1
        if (dt.datetime(db["data_prossima_stat_m"][0], db["data_prossima_stat_m"][1], db["data_prossima_stat_m"][2]) - data_oggi).days < 1 \
                and db["Cycle_monthLever"] == True:
        # if 1 == 1:

            try:
                # GRAFICO DEL PESO
                bot.send_message(a_uid, "GRAFICO PESO:")
                months = 3
                res = Grafico_peso(db["peso_bilancia"], months)
                if res:
                    fname, msg = res
                    img = open(fname, "rb")
                    msg_ = bot.send_photo(a_uid, img, caption=msg, parse_mode=pm["HTML"])
                    # db["preserved_msgIds"].append(msg_.message_id)
                    preserve_msg(msg_)
                    # db_insert("preserved_msgIds", msg_.message_id, mode="append")

                else:
                    bot.send_message(a_uid, f"Nessun pesamento negli ultimi {months} mesi")

                # BACKUP DATABASE
                bot.send_message(a_uid, "BACKUP DATABASE")
                send_db_json()

                # GRAFICO ALLENAMENTI MENSILE
                bot.send_message(a_uid, "GRAFICO MENSILE:")

                msg, img = Grafico_mensile(db, somma_numeri_in_stringa, m_to_mesi, ora_EU, actual_environment)
                msg_ = bot.send_photo(a_uid, img)
                # db_insert("preserved_msgIds", msg_.message_id, mode="append")
                preserve_msg(msg_)
                # db["preserved_msgIds"].append(msg_.message_id)
                msg_ = bot.send_message(a_uid, msg, parse_mode=pm["HTML"])
                # db_insert("preserved_msgIds", msg_.message_id, mode="append")
                preserve_msg(msg_)

                # db["preserved_msgIds"].append(msg_.message_id)

                # DB
                calendario_list = db["calendario"]
                calendario_list[1] += 1
                if calendario_list[1] == 13:
                    calendario_list[1] = 1
                    calendario_list[0] += 1

                db_insert("data_prossima_stat_m", [calendario_list[0], calendario_list[1], calendar.monthrange(calendario_list[0], calendario_list[1])[1]])
                # db["data_prossima_stat_m"] = [calendario_list[0], calendario_list[1], calendar.monthrange(calendario_list[0], calendario_list[1])[1]]
                db_insert("Cycle_monthLever", False)
                # db["Cycle_monthLever"] = False

            except Exception as exc:
                msg_ = bot.send_message(a_uid, f"C'è stato un errore nell'iterazione mensile")
                # db["preserved_msgIds"].append(msg_.message_id)
                preserve_msg(msg_)
                msg_ = traceback_message(exc)
                preserve_msg(msg_)
                # db["preserved_msgIds"].append(msg_.message_id)

        db_insert("Cycle_processLever", False) # false vuol dire che il processo è finito
        # reboot alla fine
        # if not actual_environment:
        #     with open("temporanei/db.json", "w") as file:
        #         json.dump(db, file, indent=4)

        os.execv(sys.executable, ["python"] + sys.argv)



    # RESET LEVERS A MEZZANOTTE

    elif n_ora > 0 and n_ora < sec_22:
        db["Cycle_processLever"] = True
        db["Cycle_monthLever"] = True
        db["Cycle_4settLever"] = True
        db["Cycle_dailyLever"] = True

