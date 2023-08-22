import telebot
from telebot.types import KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from funzioni import ora_EU
import psutil
import platform
import json

# Telegram setup

# 0: tempo  1: nome  2: reps  3: serie  4: peso  5: note
TOKEN = "5012282762:AAEew28I_wMj6SJUWs7980BJr_LCXGree7k"
bot = telebot.TeleBot(TOKEN)
A_UID = 305444830

blank_char = "⠀"
MD = "Markdown"
KEYBOARD_START = ReplyKeyboardMarkup(one_time_keyboard=False)
KEYBOARD_START.add("/start", "/tempi", "/database", "/archivio", "/score")


def bot_trySend(msg_text, reply_markup=None, parse_mode="HTML"):
    try:
        msg_ = bot.send_message(A_UID, msg_text, parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as exc:
        if "message is too long" in str(exc):
            msg_ = split_msg(A_UID, f"Message is too long: \n\n{msg_text}")
        else:
            msg_ = bot.send_message(A_UID, f"{msg_text}\nMarkdown error\n{exc}", parse_mode=None, reply_markup=reply_markup)

    return msg_

def addTo_botStarted(new_txt):
    ora = ora_EU(soloOre=True)
    botStarted_dict["txt"] += f"`{ora}` *\>* {new_txt}\n"
    bot.edit_message_text(text=botStarted_dict["txt"], chat_id=A_UID, message_id=botStarted_dict["msg"].message_id, parse_mode="MarkdownV2")

    print(f"{botStarted_dict['txt']}\n")

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

<b>CPU Info</b>
Physical cores: {psutil.cpu_count(logical=False)}
Total cores: {psutil.cpu_count(logical=True)}
Max Frequency: {cpufreq.max:.2f}Mhz
Min Frequency: {cpufreq.min:.2f}Mhz
Current Frequency: {cpufreq.current:.2f}Mhz
Total CPU Usage: {psutil.cpu_percent()}%

<b>Memory Information</b>
Total: {get_size(svmem.total)}
Available: {get_size(svmem.available)}
Used: {get_size(svmem.used)}
Percentage: {svmem.percent}%
"""
bot_trySend(sys_info_msg, parse_mode="HTML")


# Bot started
ora = ora_EU(soloOre=True)  # HH:mm:ss
print(f"BOT STARTED {ora}\n")

# ! disclaimer keyboards: nell'api dei bot di telegram i messaggi che hanno un reply_markup non possono essere modificati, inoltre l'unico modo per mostrare un reply_markup è
#   mandando un messaggio che la contiene
# messaggio vuoto per la keyboard
bot.send_message(chat_id=A_UID, text=blank_char+"🟢", disable_notification=True, reply_markup=KEYBOARD_START)

botStarted_dict = {"txt": f"`{ora}` *\>* Bot started\n"}
botStarted_dict["msg"] = bot.send_message(chat_id=A_UID, text=botStarted_dict["txt"], disable_notification=True, parse_mode="MarkdownV2")

if "iMac" in uname.node:
    actual_environment = False
    db_file = "db_test.json"
    copy_real_db = True
    msg_db = "using test database"

    # copiare il db vero
    if copy_real_db:
        with open("db.json", "r") as file:
            db_json = json.load(file)
        with open("db_test.json", "w") as file:
            json.dump(db_json, file, indent=4)
        print("\033[92m" + "The real database has been copied into the test database\n" + "\033[0;0m")
        msg_db = "using copy of original database"

    addTo_botStarted(f"Running in *test* environment, {msg_db}")
else:
    actual_environment = True
    db_file = "db.json"
    addTo_botStarted("Running in *real* environment")



db = json.load(open(db_file, "r"))

# IMPORTS

from creazione_immagini import Grafico_peso, Grafico_4sett, Grafico_mensile, schedaIndex_to_image, Grafico_radarScheda

import copy

import threading

import time
import datetime as dt
import calendar

import json
from funzioni import weekday_nday_month_year, schedaFname_to_title, secs_to_hours, ic, is_digit, stringaPeso_to_num

import traceback
import os

from pushover import Client
# nota: la versione 1.3 non funzionava (diceva che la richiesta per mandarenotifiche era malformata) quindi ho scaricato la 1.2.2
# oppure c'erano solo 4 versioni e quelle più recenti (non troppo vecchie da non funzinoare) non venivano installate per un problema con setuptools, quindi ho dovuto fare pip
    # install setuptools<58.0.0

# COSTANTI  #

# push notifier
CLIENT = Client("u1rztsuw5cguaeyb1y8baimqb2pw4g", api_token="a2sr9qhf5x6t2q9vfhhervbrfhq651")

# comandi generali che si possono fare sia durante l'allenamento che non
DICT_COMANDI_GEN = {
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
DICT_COMANDI_DUR = {
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

NOMI_IN_SCHEDA = {"pettobi": "Petto Bicipiti", "schiena": "Schiena", "spalletri": "Spalle Tricipiti", "altro": "Gambe", "addominali": "Addominali"}
PRE_CONSTANT = 2  # numero di secondi che vengono tolti alle pause di esercizio: sarebbe il tempo che ci si impiega ad accendere il telefono e schiacciare su /done

ALL_COMMANDS_DICT = copy.deepcopy(DICT_COMANDI_GEN)
ALL_COMMANDS_DICT.update(DICT_COMANDI_DUR)
ALL_COMMANDS_DICT = {k: k for k in ALL_COMMANDS_DICT}

NOMI_WORKOUT = ["Addominali", "Petto Bi", "Schiena", "Spalle Tri", "Altro"]
LISTA_ES_SIMPLE = ["addominali", "pettobi", "schiena", "spalletri", "altro"]
START_DICT = {"Addominali": "addominali", "Petto Bi": "pettobi", "Schiena": "schiena", "Spalle Tri": "spalletri", "Altro": "altro"}
ESERCIZI_TITLE_DICT = {"addominali": "Addominali", "pettobi": "Petto Bicipiti", "schiena": "Schiena", "spalletri": "Spalle Tricipiti", "altro": "Altro"}

M_TO_MESI = {"01": "Gen", "02": "Feb", "03": "Mar", "04": "Apr", "05": "Mag", "06": "Giu", "07": "Lug", "08": "Ago", "09": "Set", "10": "Ott", "11": "Nov", "12": "Dic"}

STRUTTURABASE_DIZALL = [[60, f"{i+1}° Esercizio", "/", 4, "/", False] for i in range(20)]  # 20 esercizi vuoti, 60 pausa di base e 4 serie di base, c'è lo stretto necessario
    # per far si che funzioni in un allenamento. False serve per indicare alla creazione messaggio scheda che è una struttura base e non un vero esercizio

TYPE_TO_DIZALL_IDX = {"Nome": 1, "Reps": 2, "Peso": 4, "Note": 5}

STR_1_6_LIST = [str(i) for i in range(1, 7)]
PRED_IDX_TO_TYPE = {0: "Nome", 1: "Reps", 2: "Peso", 3: "Note"}
DIZALL_IDX_TO_TYPE = {0: "Pausa", 1: "Nome", 2: "Reps", 3: "Serie", 4: "Peso", 5: "Note"}
LETTER_TO_TYPE = {"E": "Nome", "R": "Reps", "P": "Peso", "N": "Note"}

# KEYBOARDS
KEYBOARD_WORKOUTS = ReplyKeyboardMarkup(one_time_keyboard=True)
KEYBOARD_DONE = ReplyKeyboardMarkup(one_time_keyboard=False)
KEYBOARD_WORKOUTS.add(*NOMI_WORKOUT)
KEYBOARD_DONE.row("/done")
KEYBOARD_DONE.row("/sw", "/p", "/ese", "/n")

WORKOUTS_LISTS = {"addominali": [], "pettobi": [], "schiena": [], "spalletri": [], "altro": []}
END_SLEEP_TIME = 40 if actual_environment else 5  # secondi di sleep alla fine dell'allenamento

# tutti i valori nel g_dict non hanno bisogno di essere inizializzati perchè verranno settati strada facendo
g_dict = {}

MODIFICHE_DICT = {
    "/p": {"msg0": "Scegli l'esercizio a cui vuoi cambiare il peso",
           "db_idx": 4,
           "label": "peso"},
    "/ese": {"msg0": "Scegli il nome dell'esercizio da modificare",
             "db_idx": 1,
             "label": "nomi"},
    "/n": {"msg0": "Scegli l'esercizio a cui vuoi cambiare le note",
           "db_idx": 5,
           "label": "note"},
    "/rep": {"msg0": "Scegli l'esercizio a cui vuoi cambiare le ripetizioni",
             "db_idx": 2,
             "label": "reps"},
}

# Manipolazione database #

def db_insert(indexers, val, mode="="):
    if type(indexers) != list:
        indexers = [indexers]
    err = None
    v = db
    v_lambda = lambda v: v if len(str(v)) < 210 else f"{v[:100]}\n...\n{v[:-100]}"

    for i, indexer in enumerate(indexers):
        if type(v) == dict:
            if indexer in v:
                v = v[indexer]
            elif i != len(indexers)-1:
                err = f"KeyError: {indexer} isn't a key in {v_lambda(v)}"
                break
        elif type(v) == list:
            if type(indexer) == str:
                err = f"IndexError: using a string as a list index (idx: {indexer}, list: {v_lambda(v)})"
                break
            elif indexer < len(v):
                v = v[indexer]
            else:
                err = f"IndexError: {indexer}th index is out of range in {v_lambda(v)}"
                break
        else:
            err = f"TypeError: {indexer} was used as index for {v_lambda(v)}"
            break

    if err:
        err = "Error in db_insert function:\n" + err
        bot_trySend(msg_text=err)
    else:
        indexing_str = "".join([f"['{i}']" if type(i) == str else f"[{i}]" for i in indexers])
        ic(indexing_str)
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

# key = "aumenti_msgScheda"
# if key not in db.keys():
#     db[key] = {"addominali": [], "pettobi": [], "schiena": [], "spalletri": [], "altro": []}


"""FUNZIONI"""

# FUNZIONI ALLENAMENTO

def get_dataSchedaAttuale():
    date_schede = list(db["schede"].keys())
    date_schede.sort()
    return date_schede[-1]

# def escape_dizAll(diz_all_):
#     """ otteniamo un dizall dove i nomi ese, reps, peso e note sono compatibili con markdown v2 """
#     for workout, lists in diz_all_.items():
#         for i, list_ in enumerate(lists):
#             diz_all_[workout][i][1] = md_v2_replace(diz_all_[workout][i][1])
#             diz_all_[workout][i][2] = md_v2_replace(diz_all_[workout][i][2])
#             diz_all_[workout][i][4] = md_v2_replace(diz_all_[workout][i][4])
#             diz_all_[workout][i][5] = md_v2_replace(diz_all_[workout][i][5])
#
#     return diz_all_

def eseSerie_to_timerMsg(diz_all, workout, ese, serie):
    quote = '"'  # usando \" replit aveva problemi nell'interpetazione del codice per qualche motivo
    timer_msg = f"<i>{diz_all[workout][ese][1]}</i>\n" \
                f"<b>{serie + 1} serie su {diz_all[workout][ese][3]}</b>\n" \
                f"<b>R:</b> {diz_all[workout][ese][2]}\n" \
                f"<b>P:</b> {diz_all[workout][ese][4]}\n" \
                f"<b>Pausa:</b> <code>{diz_all[workout][ese][0]}{quote}</code>" \
                f"\nNote: <i>{diz_all[workout][ese][5]}</i>".replace("\nNote: <i>None</i>", "").replace("\nNote: <i>False</i>", "")

    return timer_msg

# FUNZIONI TEMPO

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

# FUNZIONI GENERICHE

def reverse_dict(dict):
    keys_list = list(dict.keys())
    values_list = list(dict.values())

    return {values_list[i]: keys_list[i] for i in range(len(keys_list))}

def flatten_list(lista):
    return [i for sublist in lista for i in sublist]

def send_db_json():
    """ funzione per inviare il json del db """
    if actual_environment:
        date = weekday_nday_month_year(ora_EU(), year=True)
        fname = f"db/backup-db_{date}.json"
        with open(fname, "w") as file:
            json.dump(db, file, indent=4)

        sendable_cloned_db = open(fname)
        msg_ = bot.send_document(A_UID, sendable_cloned_db)
        preserve_msg(msg_)
        # db_insert("preserved_msgIds", msg_.message_id, mode="append")
        # db["preserved_msgIds"].append(msg_.message_id)

        return open(fname)

def somma_numeri_in_stringa(stringa):
    # todo controlalre se è rimpiazzabile dalla versione migliore che fa anche la media
    """ funzione che viene usata per calcolare la somma dei numeri nella stringa """
    # queste operazioni di replacement ci servono per trattare in modo diverso i ", " e ",", visto che una separa il peso e l'altra è un numero con la virgola
    stringa = stringa.replace(", ", "$$$")
    stringa = stringa.replace(",", ".")
    stringa = stringa.replace("$$$", ", ")

    num_string = ""
    num_lever = False

    sum_list = []

    for char in stringa:
        # se il carattere è un numero
        if char in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]:
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

def remove_html(string):
    string = str(string)
    return string.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", "")

# FUNZIONI TELEGRAM

def split_msg(uid, string, parse_mode=None):
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
        buttons_list.append(InlineKeyboardButton(text=text, callback_data=raw_buttons["callbacks"][i]))

    keyboard = InlineKeyboardMarkup(row_width=row_width)
    keyboard.add(*buttons_list)

    return keyboard

def bot_tryDelete(uid, message_id):
    try:
        if message_id not in db["preserved_msgIds"]:
            bot.delete_message(uid, message_id)
    except:
        pass

def traceback_message(exc):
    bot.send_message(A_UID, f"ERRORE: {exc} \nRicordarsi di salvare il traceback se lo si vuole visualizzare in futuro")
    fname = "temporanei/latest_traceback.txt"
    with open(fname, 'w') as f:
        f.write(str(traceback.format_exc()))
    file = open(fname, "rb")
    msg_ = bot.send_document(A_UID, file)

    return msg_


# FUNZIONI AI

def idx_to_oneHot(idx, labels_num):
    one_hot = [0. for _ in range(labels_num-1)]
    one_hot.insert(idx, 1.)

def start_training_procedure():
    try:
        bot.send_message(A_UID, "Allenamento del modello in corso...")

        from training import train_model
        summary_msg, loaded_files = train_model("msg labelling", "hpo_model", db)

        bot.send_message(A_UID, summary_msg, parse_mode="HTML")
        [bot.send_document(A_UID, file) for file in loaded_files]

    except Exception as exc:
        msg_ = bot.send_message(A_UID, text="L'allenamento del modello non è andato a buon termine:")
        preserve_msg(msg_)
        msg_ = traceback_message(exc)
        preserve_msg(msg_)

# TRAINING

class Training:
    # INIT

    def __init__(self):
        cd = copy.deepcopy
        self.serie_in_ese_list = {}
        """ serie_in_ese_list = {"addominali": [[serie_es-1, serie_es-1, serie_es-1], [serie_es-2, ...] ...] """  # serie_es-N è sempre 0
        self.num_ese = {}
        """ numero totale di esercizi per workout """

        self.flat_timer_msg_list = {}
        self.flat_pause_list = {}

        self.startup_db = cd(db)

        self.usermode = "none"
        self.sequences_usermode = "none"
        self.fineTimer_msgId = 0
        self.schedaPrev_msgId = 0
        self.restart_msgId = 0

        self.strikethrough_list_msg = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" "", "", ""]

        self.switch_list = cd(WORKOUTS_LISTS)
        """ switch_list all'inizio: [nome_ese_1, nome_ese_2, ...] 
            switch_list dopo aver finito un esecizio: ["", nome_ese_2, ...] """
        self.switch_list_quotes = cd(WORKOUTS_LISTS)  # versione con solo "" di switch list

        # prev_tempo_building: andrà a sostituire il prev_tempo più vecchio, che a sua volta era un prev_tempo_building, db["prev_tempo"] è formato da 3db db_prev_tempo building
        self.prev_tempo_building = cd(WORKOUTS_LISTS)
        self.peso_ese_list = cd(WORKOUTS_LISTS)

        self.counter = 0
        self.count_serie = cd(WORKOUTS_LISTS)
        """ count_serie[workout] = [[0, 1, 2, 3], [4, 5, 6, 7, 8], ...]  # cambia solo con /switch e quando cambiamo il numero di serie durante la creazione scheda"""
        self.count_serie_flat = {}
        """
        prima del primo /done:  count_serie_flat[workout] =    [0, 1, 2, 3, 4, ...]
        dopo il primo /done:    count_serie_flat[workout] =    [1, 2, 3, 4, ...]
        dopo il secondo /done:  count_serie_flat[workout] =    [2, 3, 4, ...]
        """
        self.counter_dict = {}
        """ counter_dict = {'addominali': [4, 8, 11], 'pettobi': [...], ...}, vuol dire che addominali aveva il  primo ese con 4 serie, il secondo con 4, terzo con 3 """
        self.completed_counters = {"addominali": [0], "pettobi": [0], "schiena": [0], "spalletri": [0], "altro": [0]}  # vengono aggiunti i numeri dei counter completati, lo 0,
        #   cio la prima serie è già segnata come completata e bisogna dare il via (/done) per avviare il timer della seconda

        self.current_ese_idx = 0  # idx dell'esercizio a cui siamo nell'allenamento

        # BOOLS
        self.send_radar_graph = False
        self.send_grafico_peso = False
        self.creazione_scheda = False
        self.run_timer = False
        self.switched_first_ese = False
        self.notime = False
        self.restart = False
        self.skip = False
        self.selected_workout = False
        self.callback_antiSpam = False

        self.creazioneScheda_undo = {"cambioScheda_msgId": None}  # inizializziamo con None perchè ci serve per far sapere al bot se sono già stati mandati dei messaggi di
        #   compilazione o meno

        # time prediction data di questo workout, verrà aggiunta al db OBIETTIVO: PREVEDERE I SECONDI RESTANTI
        self.TP_data = dict(timedate=None, workout=None, n_workout=0,  # info
                            S_nomi=[], S_reps=[], S_pause=[], S_serie=[],  # scheda
                            ultimi_allenamenti=[], current=[])  # current = w_sec_taken di questo workout

        self.prev_tempo = {}
        for w in WORKOUTS_LISTS.keys():
            self.prev_tempo[w] = self.startup_db["w_sec_taken"][w][:3]

            diff = len(self.prev_tempo[w]) - 3
            if diff < 0 and self.prev_tempo[w]:
                [self.prev_tempo[w].append(self.prev_tempo[w][-1]) for _ in range(abs(diff))]
        self.avg_adj_prev_tempo = cd(WORKOUTS_LISTS)  # avg: è la media degli ultimi 3 allenamenti per ogni esercizio, adj: adjusted: quando facciamo un /done viene inserito in
        #   [ese] il tempo che ci abbiamo messo, quindi usa il tempo che ci abbiamo messo in questo allenamento per dirci quanto tempo abbiamo messo dall'inizio, e per la previsione
        #   usa la media degli scorsi allenamenti

        self.data_scheda = get_dataSchedaAttuale()
        self.diz_all = self.startup_db["schede"][self.data_scheda]
        self.active_workouts = [workout for workout, excercises in self.diz_all.items() if excercises != [] or excercises != STRUTTURABASE_DIZALL]

        for workout in self.active_workouts:
            self.adapt_to_workout(workout)  # questo vuol dire che creiamo dizionari e liste solo per i gruppi muscolari che esistono nella scheda
        self.ese_names_keyb = {w: ReplyKeyboardMarkup() for w in self.active_workouts}
        [self.ese_names_keyb[w].add(*[self.diz_all[w][ese][1] for ese in range(self.num_ese[w])]) for w in self.active_workouts]

        self.switch_list_full = cd(self.switch_list)  # switch list è stata completata in adapt to workout
        """ switch_list_full: [nome_ese_1, nome_ese_2, ...] """  # e rimane sempre uguale
        self.count_serie_full = cd(self.count_serie)

    # ADAPTING

    def adapt_to_newSerie(self, w):
        """ questa funzioe viene chiamata nella creazione iniziale dei dizionari e quando il numero di serie cambia e ricrea alcuni dizionari usando le serie nuove """

        self.serie_in_ese_list[w] = [ese[3] for ese in self.diz_all[w]]
        self.num_ese[w] = len(self.diz_all[w])

        self.counter_dict[w] = []

        self.flat_timer_msg_list[w] = []
        self.flat_pause_list[w] = []

        serie_flat_counter = 0  # 0 1 2 3 4 5 6 7 ...

        for i_ese, ese in enumerate(self.serie_in_ese_list[w]):
            self.counter_dict[w].append(sum([self.diz_all[w][i][3] for i in range(i_ese+1)]))
            self.count_serie[w].append([])

            for serie in range(ese):
                self.flat_timer_msg_list[w].append(eseSerie_to_timerMsg(self.diz_all, w, i_ese, serie))

                self.flat_pause_list[w].append(self.diz_all[w][i_ese][0])

                self.count_serie[w][-1].append(serie_flat_counter)
                serie_flat_counter += 1

    def adapt_to_workout(self, w):  # prima era inizializzazione()

        self.adapt_to_newSerie(w)
        serie_count = 0

        # itera per ogni esercizio
        for i_ese, ese in enumerate(self.serie_in_ese_list[w]):

            # creazione delle keyboard
            self.peso_ese_list[w].append(f"{self.diz_all[w][i_ese][4]}  {self.diz_all[w][i_ese][1]}")  # peso e nome
            self.switch_list[w].append(self.diz_all[w][i_ese][1])
            self.switch_list_quotes[w].append("")

            # itera ogni serie
            for serie in range(ese):
                self.prev_tempo_building[w].append(0)  # il prev tempo è formato da 0 che non vengono usati e gli idxs man mano vengono cambiati con il tempo impiegto

                if self.prev_tempo[w]:  # se non è []
                    # serve in caso ci fossero errori tra prev_tempo nel db e numero di serie totali, di norma non dovrebbe succedere ma è successo ed è un easy fix
                    if len(db["w_sec_taken"][w][0]) < serie_count+1:
                        [db_insert(["w_sec_taken", w, i], 100, mode="append") for i in range(len(db["w_sec_taken"][w]))]

                    self.avg_adj_prev_tempo[w].append(sum([self.prev_tempo[w][i][serie_count] for i in range(3)]) / 3)

                serie_count += 1

            # ese_names_keyb[w].add(self.esc_dizAll[w][ese][1])

    # MESSAGGI SCHEDA

    def crea_messaggioScheda(self):
        """ questa funzione crea il messaggio di scheda """
        w = self.usermode

        msg_scheda_tot = f"<code>{blank_char}    </code><b>{NOMI_IN_SCHEDA[w].upper()}</b>\n\n"
        circlepos = 0
        circle_lever = True

        # AUMENTI PESO IN SCHEDA
        cloned_aum_msgScheda = [{k: v for k, v in aum_dict.items()} for aum_dict in db["aumenti_msgScheda"][w]]
        aums = [i | {"w_distance": self.startup_db["workout_count"][w] - i["w_count"]} for i in cloned_aum_msgScheda]
        aums = [i for i in aums if i["w_distance"] < 3]  # 0: questo w, 1: scorso w, 2: w prima dello scorso

        ese_idxs = [i["ese_idx"] for i in aums]
        aums_dict = {idx: [] for idx in set(ese_idxs)}
        for i in aums:
            aums_dict[i["ese_idx"]].append(i)
        aums_dict = {idx: i[-1] for idx, i in aums_dict.items()}  # l'ultimo item dovrebbe è il più recente, quindi lo scegliamo

        # ITERAZIONE AGGIUNTA ESERCIZI E PALLINO BIANCO
        for ese in range(len(self.counter_dict[w])):
            # strikethrough
            if self.switch_list[w][ese] == "":  # se l'esercizio è già stato completato
                self.strikethrough_list_msg[ese] = "<strike>"

            if self.counter < self.counter_dict[w][ese] and circle_lever:
                circlepos = ese  # int che indica a quale esercizio mettere il cerchio
                circle_lever = False  # una volta trovato non ci ritorna su questo punto di codice

        for ese in range(len(self.serie_in_ese_list[w])):
            # ... in caso di creazione scheda
            if ese == 0:  # serve per evitare di fare il prossimo if statement se l'index è 0 (ci sarebbe index error)
                pass
            elif self.diz_all[w][ese-1][5] is not False:  # se le note sono False, quindi se è una struttura base db
                pass
            else:  # se l'esercizio prima di questo era una struttura base
                msg_scheda_tot += "<code>    </code>..."
                break

            # questa parte di codice non viene raggiunta se si finisce nell'else perchè c'è il break
            pausa = str(self.diz_all[w][ese][0])
            pausa_padding = "  " if len(pausa) == 2 else " "  # questo codice serve per padding (mantenere la lunghezza uguale anche quando ci sono 3 numeri al posto che 2)
            pau = f'{pausa}"{pausa_padding}'

            nome_ese = self.diz_all[w][ese][1]
            circle = "⚪️️️ " if ese == circlepos else ""
            slm1 = self.strikethrough_list_msg[ese]
            slm2 = f"{self.strikethrough_list_msg[ese][:1]}/{self.strikethrough_list_msg[ese][1:]}" if slm1 == "<strike>" else ""

            aum_old = ""
            aum_emoji = " "*5
            if ese in aums_dict:
                emojis = {0: "💹", 1: "✳️", 2: "*️⃣"}
                aum_emoji = f"  {emojis[aums_dict[ese]['w_distance']]} "
                aum_old = f"\n<code>     </code><strike>{aums_dict[ese]['peso']}</strike>"

            msg_scheda_tot += f"<code>{pau}</code>" +       f"{circle}{slm1}<i>{nome_ese}</i>{slm2}\n" \
                              f"<code>     </code>" +       f"S: {self.diz_all[w][ese][3]}   R: {self.diz_all[w][ese][2]}\n" \
                              f"<code>{aum_emoji}</code>" + f"{self.diz_all[w][ese][4]}{aum_old}\n\n"

        return msg_scheda_tot

    def workout_finish_secs(self, w, addo_bool):
        """
        funziona che ci dice a che ora finiremo l'allenamento in base a quando abbiamo iniziato e dopo quanti secondi è previsto che finiamo
        ci dice anche quanti minuti mancano all'ora che viene indicata come fine dell'allenamento
        """
        seconds = sum(T.avg_adj_prev_tempo[w])
        # fine = ora in cui finiremo l'allenamento in secondi, è formato dai secondi che corrispondono all'ora di quando abbiamo iniziato l'esercizio + i secondi che ci metteremo a
        #   finire (secondo le previsioni)
        fine = sum([a * b for a, b in
                    zip([3600, 60, 1], map(int, g_dict["inizioAll_HH_MM_SS"].split(":")))]) + seconds

        ora = ora_EU(soloOre=True)
        ora_in_secondi = sum([a * b for a, b in zip([3600, 60, 1], map(int, ora.split(":")))])

        if not addo_bool:
            return f"<code>{str(dt.timedelta(seconds=fine))[0:5]}</code>, <b>-{secs_to_hours(abs(fine - ora_in_secondi))}</b>"
        else:
            return f"<code>{str(dt.timedelta(seconds= fine))[0:5]}</code>, <b>-{secs_to_hours(abs(fine - ora_in_secondi))}</b>\n" \
                   f"🧭 Con addominali: <code>{str(dt.timedelta(seconds= fine + sum(self.avg_adj_prev_tempo['addominali'])))[0:5]}</code>, " \
                   f"<b>-{secs_to_hours(fine + sum(self.avg_adj_prev_tempo['addominali']) - ora_in_secondi)}</b>"

    def invia_messaggioScheda(self, send=False):
        """ ottiene e manda sia la scheda di allenamento che la previsione del tempo """
        w = self.usermode

        addo_bool = False if w == "addominali" else True  # ci viene detto quanto ci mettiamo a fare anche gli addominali solo se stiamo facendo un esercizio diverso da addominali
        fine_prevista = f"🧭 Fine prevista: {self.workout_finish_secs(w, addo_bool)}" if self.creazione_scheda == False else ""

        msg_scheda = self.crea_messaggioScheda()
        msg_schedaPrev = fine_prevista
        if send:
            # questi due messaggi devono essere separati, visto che solo msg_scheda subisce degli edit_message, e ci serve un messaggio che non viene modificato per usare le keyboards
            msg_scheda = bot_trySend(msg_scheda)
            if fine_prevista != "":
                msg_schedaPrev = bot_trySend(msg_schedaPrev, reply_markup=KEYBOARD_DONE)

        return msg_scheda, msg_schedaPrev


    # SELECT & START WORKOUT

    def select_workout(self, msg_id):
        if self.restart == False:

            if not self.selected_workout:
                bot_tryDelete(A_UID, self.restart_msgId)

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
                        ultimi_allenamenti_workouts[giorni_fa].append(NOMI_IN_SCHEDA[lista_[1]])
                    else:
                        break

                msg_start = f'<b>Inizio scheda attuale:</b> <code> {int(self.data_scheda[-2:])} {M_TO_MESI[self.data_scheda[-5:-3]]}</code>\n\n'
                msg_start += "<b>Ultimi allenamenti:</b>\n    "

                ultimi_allenamenti_strings = []
                for giorni_fa, nomi_w in ultimi_allenamenti_workouts.items():
                    string = ""
                    for i, nome in enumerate(nomi_w):
                        if len(nomi_w) - (i+1) == 0:  # se non seguono altri nomi per questo giorno
                            string += f"<b>{nome}</b>\n"
                        else:
                            string += f"<b>{nome}</b>, "
                    string += f"        <code>{weekday_nday_month_year(oggi_mezzanotte_iso - dt.timedelta(days=giorni_fa), weekday=True)}</code>, <i>{giorni_fa} giorni fa</i>\n    "
                    ultimi_allenamenti_strings.append(string)
                ultimi_allenamenti_strings.reverse()  # reversiamo così che gli ultimi giorni saranno primi

                msg_start += "".join(stringa for stringa in ultimi_allenamenti_strings)
                msg_start = msg_start.replace(" <i>0 giorni fa", " <i>Oggi").replace(" <i>1 giorni fa", " <i>Ieri")


                raw_buttons = dict(texts=[val for _, val, in ESERCIZI_TITLE_DICT.items()],  # tutti gli esercizi
                                   callbacks=[f"WORK{key}"  for key, _, in ESERCIZI_TITLE_DICT.items()])
                buttons = inline_buttons_creator(raw_buttons, 2)

                msg_ = bot_trySend(msg_start, reply_markup=buttons)

                g_dict["sceltaAll_msgId"] = msg_.message_id
                self.counter = 0
                g_dict["start_msgId"] = msg_id
                self.selected_workout = True
                self.callback_antiSpam = False

            else:
                bot_tryDelete(A_UID, msg_id)

        # se lever_restart è vera, la mettiamo come falsa e riavviamo il bot, quindi torna falsa
        else:
            msg_ = bot_trySend("Restarting...")
            globals()["T"] = Training()
            T.restart_msgId = msg_.message_id
            T.select_workout(msg_id)

    def start_workout(self, w):
        bot.edit_message_reply_markup(A_UID, g_dict["sceltaAll_msgId"], reply_markup=None)

        self.usermode = w
        g_dict["inizioAll_HH_MM_SS"] = ora_EU(soloOre=True)
        g_dict["inizioAll_secs"] = time.perf_counter()
        self.restart = True

        # ORDINARIO
        if db["schede"][self.data_scheda][w] != STRUTTURABASE_DIZALL:  # nel caso in cui il w è già stato compilato
            n_ese = len(self.diz_all[w])
            msg_scheda, msg_schedaPrev = self.invia_messaggioScheda(send=True)
            g_dict["scheda_msgId"], _ = msg_scheda.message_id, msg_schedaPrev.message_id  # il messaggio della previsione del tempo non viene salvato (_) e in questo modo non viene
                # eliminato e rimane la prima previsione del bot che alla fine si può confrontare con il risultato finale

            self.TP_data = dict(timedate=ora_EU(isoformat=True), workout=w, n_workout=db["workout_count"][w]+1,  # info
                                S_pause=[self.diz_all[w][i][0] for i in range(n_ese)],
                                S_nomi =[self.diz_all[w][i][1] for i in range(n_ese)],
                                S_reps =[self.diz_all[w][i][2] for i in range(n_ese)],
                                S_serie=[self.diz_all[w][i][3] for i in range(n_ese)],
                                ultimi_allenamenti=self.startup_db["w_sec_taken"][w],
                                current=None)

        # NUOVA SCHEDA
        else:
            self.creazione_scheda = True

            # modello DL
            msg_ = bot.send_message(A_UID, "Importazione dell'intelligenza artificiale...")
            msg_clessidra = bot.send_message(A_UID, text="⏳")

            global tf
            import tensorflow as tf

            model_name = "hpo_model"
            g_dict["msgLabelling_model"] = tf.keras.models.load_model(f"msg labelling/{model_name}")
            plot_name = f"immagini/model_plot.png"
            # v- NON funziona su replit
            # tf.keras.utils.plot_model(global_dict["msgLabelling_model"], to_file=plot_name, show_shapes=True, show_layer_activations=True)
            bot.send_photo(A_UID, photo=open(plot_name, "rb"), caption=f"🤖 Modello in uso: {db['NNs_names'][model_name]['name']} V{db['NNs_names'][model_name]['version']}")

            bot.delete_message(A_UID, msg_.message_id)
            bot.delete_message(A_UID, msg_clessidra.message_id)

            # stats per radar graph
            db_insert(["workout_count", w], 0)
            db_insert(["data_2volte", w], None)
            db_insert(["dizAll_2volte", w], None)

            # messaggio di aiuto e messaggio scheda
            bot.send_message(A_UID, text="Info sulla creazione scheda:\n"
                                         "•  Format per le serie: 'N', (N: num da 1 a 6)\n"
                                         "•  Format per la pausa: 'NN\"' (N: num)\n"
                                         "•  Nome esercizio, reps, peso e note sono gestite dal neural network. nel caso in cui si voglia usare un numero di reps che va in "
                                         "conflitto con le regole per le serie, scrivere \"N reps\" e verrà preso solo il numero N\n"
                                         "•  All'ultima serie dell'ultimo esercizio, dopo aver fatto tutte le modifiche, fare /fine, se non si fa questo ultimo passaggio "
                                         "nessuna modifica viene applicata",
                             reply_markup=KEYBOARD_DONE)

            # db["data_cambioscheda"].append(ora_EU(isoformat=True))
            db_insert("data_cambioscheda", ora_EU(isoformat=True), mode="append")
            msg_scheda = bot_trySend(f"{self.crea_messaggioScheda()}")
            g_dict["scheda_msgId"] = msg_scheda.message_id

    # TIMER
    def timer(self, msg):
        w = self.usermode
        refresh_rate = 1
        num_ora_inizio = precise_numero_ora()
        lever_10_sec = True

        self.count_serie_flat[w] = flatten_list(self.count_serie[w])
        [self.count_serie_flat[w].remove(counter) for counter in self.completed_counters[w]]  # rimozione dei counter già completati
        self.counter = self.count_serie_flat[w][0]

        # ottenimento di i_ese
        for i_ese, max_counter in enumerate(
                ([0] + self.counter_dict[w])[1:]):  # inseriamo uno 0 all'inizio e partiamo dal secondo elemento per non avere problemi di iterazione
            if self.counter < max_counter and self.counter >= self.counter_dict[w][
                i_ese - 1]:  # se il counter è minore del counter massimo per l'esercizio e maggiore del coutner
                # massimo di prima
                self.current_ese_idx = i_ese
                break

        # messaggio scheda
        msg_scheda, msg_schedaPrev = self.invia_messaggioScheda(send=True)
        self.scheda_msgId = msg_scheda.message_id
        if type(msg_schedaPrev) != str:
            self.schedaPrev_msgId = msg_schedaPrev.message_id

        # messaggio con timer
        msg_timer = bot_trySend(f"{self.flat_timer_msg_list[w][self.counter]} <b>{self.flat_pause_list[w][self.counter]}</b>")

        refreshed_num_ora = 0
        secs_after = time.perf_counter()
        self.run_timer = True

        # DURANTE TUTTO L'ALLENAMENTO, FINO ALLA FINE
        while True:
            numOra_fineTimer = num_ora_inizio + self.flat_pause_list[w][self.counter] - PRE_CONSTANT

            # DURANTE TIMER
            if self.run_timer == True and refreshed_num_ora + refresh_rate < numOra_fineTimer and self.skip == False:

                refreshed_num_ora = precise_numero_ora()
                timer_piece = f"🟡 <b>Timer</b>: <code>{int(numOra_fineTimer - refreshed_num_ora)}\"</code>"

                # 10 SECONDS WARNING, se mancano 10 secondi:
                if lever_10_sec == True and refreshed_num_ora > numOra_fineTimer - 9:
                    # todo togliere <code> <i> ecc dalla notifica
                    noti = remove_html(self.flat_timer_msg_list[w][self.counter])
                    CLIENT.send_message(f"🟡 8 secondi restanti\n{noti}", title="Timer")
                    lever_10_sec = False

                # TIMER MESSAGE EDITING, se siamo ancora < di numOra_fineTimer:
                if (refreshed_num_ora + refresh_rate + 0.1) < numOra_fineTimer:

                    try:
                        bot.edit_message_text(chat_id=msg.chat.id, text=f"{self.flat_timer_msg_list[w][self.counter]}\n\n{timer_piece}",
                                              message_id=msg_timer.message_id, parse_mode="HTML")
                    except:
                        pass

                    secs_before = time.perf_counter()
                    secs_taken = secs_before - secs_after

                    time.sleep(
                        max(refresh_rate - secs_taken, 0.01))  # refresh_rate - secs_taken serve epr dormire precisamente 1 secondo, max() serve eper evitare numeri negativi
                    secs_after = time.perf_counter()

                # se invece refreshed_num_ora + refresh_rate è maggiore fermiamoci per il tempo rimanente e poi passiamo a FINE TIMER
                else:
                    refreshed_num_ora = precise_numero_ora()
                    time.sleep(max(numOra_fineTimer - refreshed_num_ora, 0.01))  # 0.1 è per sicurezza in modo da non fare mai sleep di un tempo negativo

            # FINE TIMER
            else:
                self.run_timer = False
                bot_tryDelete(msg.chat.id, msg_timer.message_id)

                self.completed_counters[w].append(self.counter)
                # se era l'ultima serie
                for ese in range(len(self.counter_dict[w])):
                    if self.counter + 1 == self.counter_dict[w][ese]:
                        self.switch_list[w][ese] = ""

                last_set = self.switch_list[w] == self.switch_list_quotes[w]

                # messaggio e notifica
                if self.skip == False or last_set:
                    noti = remove_html(self.flat_timer_msg_list[w][self.counter])
                    CLIENT.send_message(f"🔴 Fine timer\n{noti}", title="Timer")

                    msg_fineTimer = bot_trySend(f"{self.flat_timer_msg_list[w][self.counter]}\n\n🔴 <b>Timer finito</b>", KEYBOARD_DONE)
                    self.fineTimer_msgId = msg_fineTimer.message_id

                    if last_set:
                        time.sleep(END_SLEEP_TIME)
                        bot_tryDelete(A_UID, msg_fineTimer.message_id)

                        self.end_workout()
                        self.saveSend_TP()

                elif self.skip == True:
                    self.done(msg)
                    self.skip = False

                break  # interrompe while True

    # DONE & SWITCH

    def done(self, msg):
        w = self.usermode
        uid, msg_id = msg.chat.id, msg.message_id

        if self.sequences_usermode != "none":
            self.sequences_usermode = "none"
            bot_tryDelete(uid, msg_id-1)

        bot_tryDelete(uid, msg_id)
        bot_tryDelete(uid, g_dict["scheda_msgId"])
        if self.counter != 0 and not self.creazione_scheda:
            bot_tryDelete(uid, self.schedaPrev_msgId)
        bot_tryDelete(uid, self.fineTimer_msgId)

        # PREV TEMPO

        # primo /done
        if self.counter == 0:
            g_dict["continuo1"] = time.perf_counter()
            self.prev_tempo_building[w][self.counter] = g_dict["continuo1"]-g_dict["inizioAll_secs"]

        # tutti i /done dopo il primo
        else:
            g_dict["continuo2"] = time.perf_counter()
            self.prev_tempo_building[w][self.counter] = g_dict["continuo2"]-g_dict["continuo1"]
            g_dict["continuo1"] = time.perf_counter()

        if len(self.avg_adj_prev_tempo[w]) > self.counter:  # condizione ordinaria
            self.avg_adj_prev_tempo[w][self.counter] = self.prev_tempo_building[w][self.counter]
        else:  # questo caso serve ad evitare bug nel caso in cui self.avg_adj_prev_tempo è più piccolo del prev tempo build
            self.avg_adj_prev_tempo[w].append(self.prev_tempo_building[w][self.counter])

        t = threading.Thread(target=self.timer, args=[msg])
        t.start()

    def switch_ese(self, switch_to, msg_id):
        w = self.usermode
        index_sw = self.switch_list_full[w].index(switch_to)

        switched_counters = self.count_serie_full[w][index_sw]  # es: [4, 5, 6]
        self.count_serie[w].remove(switched_counters)
        self.count_serie[w].insert(0, switched_counters)  # li rimettiamo ad index 0, quindi saranno i prossimi counter che ci appariranno finito l'esercizio

        self.sequences_usermode = "none"

        # se almeno un esecizio è stato completato ma il primo esercizio non è stato completato, quindi se abbiamo fatto /sw appena iniziato il workout
        if self.switch_list[w][0] == self.switch_list_full[w][0] and not self.switched_first_ese:
            self.completed_counters[w].remove(0)  # rimozione del primo counter visto che non è stato compleato
            self.completed_counters[w].append(switched_counters[0])  # segnamo la prima serie dell'esercizio cambiato come completata
            self.switched_first_ese = True

        bot_tryDelete(A_UID, msg_id)
        bot_tryDelete(A_UID, g_dict["switch_msgId"])

    # FINE WORKOUT, FINE CREAZIONE SCHEDA, SEND TP DATA

    def end_workout(self, score_w=None):
        w = score_w if score_w else self.usermode
        """ funzione che si occupa di fare tutte le scritture nel db e del messaggio alla fine di un allenamento """
        ora_eu_iso = ora_EU(isoformat=True)
        fine_str = ora_EU(soloOre=True)

        # messaggio allenamento finito e aggiunta di questa fine allenamento nel db
        msg_fine = ""
        for i, fine_es in enumerate(reversed(db["fine_allenamenti"][w])):
            # primi 3 esercizi passati
            if i != 0:
                msg_fine += fine_es + "\n\n"
            else:
                pass

        # caso in cui c'è stato un allenamento con /start /done fino alla fine
        if not score_w:
            fine = time.perf_counter()
            tempo_imp = secs_to_hours(fine-g_dict["inizioAll_secs"], False)

            msg_oggi = f"Tempo impiegato: <b>{tempo_imp}</b>\nDalle {g_dict['inizioAll_HH_MM_SS'][:-3]} alle {fine_str[:-3]}"

            # salvataggio in db di "w_sec_taken"
            if self.notime == False:
                g_dict["continuo2"] = time.perf_counter()
                self.prev_tempo_building[w][self.counter] = self.prev_tempo_building[w][self.counter - 1]  # facciamo in modo che la durata dell'ultimo
                    # esercizio sia uguale a quella del penultimo
                self.prev_tempo_building[w] = [i for i in self.prev_tempo_building[w] if i != 0]

                db_insert(["w_sec_taken", w], self.prev_tempo_building[w], mode="append")

            else:  # se invece abbiamo usato /notime lo scriviamo nel msg_oggi
                msg_oggi += " (/notime)"

        else:
            msg_oggi = f"Tempo impiegato: <b>N/A</b>\nFine: {fine_str}"

        # parte di saving dizall 2 volte e conteggio w
        db_insert(["workout_count", w], 1, mode="+=")

        if db["workout_count"][w] == 2:
            db_insert(["data_2volte", w], ora_eu_iso)
            # db["data_2volte"][w] = ora_eu_iso

            db_insert(["dizAll_2volte", w], copy.deepcopy(db["schede"][self.data_scheda][w]))

        # msg
        msg_ = bot.send_message(chat_id=A_UID, text= f"🔴 Allenamento finito: <b>{ESERCIZI_TITLE_DICT[w]}:</b> ({db['workout_count'][w]}ª volta)\n\n"
                                                   f"{msg_fine}<code>Oggi</code>\n{msg_oggi}",
                                reply_markup=KEYBOARD_START, parse_mode="HTML")
        db_insert("preserved_msgIds", msg_.message_id, mode="append")
        # db["preserved_msgIds"].append(msg_.message_id)

        # eliminazione messaggi
        for messaggio in range(g_dict["start_msgId"], msg_.message_id):
            bot_tryDelete(A_UID, messaggio)

        db_insert(["fine_allenamenti", w], [db["fine_allenamenti"][w][i] for i in [-1,0,1,2]])
        str_date = f"{int(ora_eu_iso[8:10])} {M_TO_MESI[ora_eu_iso[5:7]]}"
        db_insert(["fine_allenamenti", w, 0], f"<code>{str_date}</code>\n{msg_oggi}")

        # ultimo allenamento
        db_insert("allenamenti", [ora_EU(isoformat=True), w], mode="append")

        self.sequences_usermode = "none"

    def saveSend_TP(self):
        w = self.usermode
        self.TP_data["current"] = self.prev_tempo_building[w]
        db_insert("TP_data", self.TP_data, mode="append")

        fname = "temporanei/tp_data.json"
        with open(fname, "w") as file:
            json.dump(self.TP_data, file, indent=4)

        bot.send_document(A_UID, open(fname, "rb"))

    def fine_creazione_scheda(self):
        w = self.usermode
        db["schede"][self.data_scheda][w] = [ese for ese in self.diz_all[w] if ese[5] != False]  # mettiamo nel db tutti gli esercizi di
            # diz_all che non hanno le note come "False", cioè che indicano che l'esercizio è vuoto e struttura base

        # nuovi dati di training
        for ese in db["schede"][self.data_scheda][w]:
            for i in [1, 2, 4, 5]:  # solo gli index a cui viene applicato il ML
                if i == 1 and "° Esercizio" not in ese[i]:
                    db_insert("new_training_data", [ese[i], "nomi"], mode="append")
                elif i == 2 and ese[i] != "/":
                    db_insert("new_training_data", [ese[i], "reps"], mode="append")
                elif i == 4 and ese[i] != "/":
                    db_insert("new_training_data", [ese[i], "peso"], mode="append")
                elif i == 5 and ese[i] != None:
                    db_insert("new_training_data", [ese[i], "note"], mode="append")

        self.end_workout()
        self.saveSend_TP()

    # MODIFICATIONS & COMPILE

    def apply_modification(self, text, i_ese, i_column, creazione_scheda=False):
        """ questa funzione viene usata quando si modifica qualcosa nel db, ad esempio quando si fa /p o si stra creando una nuova scheda. quello che fa è attuare le modifiche nel
        db, in diz_all, esc_dizall e in flat_timer_msg """
        w = self.usermode

        # se sono peso cambia da k a kg, se sono nomi maiuscola all'inizio
        text = text.replace("k", "㎏") if i_column == 4 else text
        text = text[0].upper() + text[1:] if i_column == 1 else text  # .capitalize() toglieva l'upper da tutto e lo metteva solo all'inizio

        # todo controllare creazione scheda e come e quando viene applicata questa funzione
        if not creazione_scheda:  # se siamo in fase di creaizone scheda non viene applicata al database
            db_insert(["schede", self.data_scheda, w, i_ese, i_column], text)

        if creazione_scheda:
            self.creazioneScheda_undo["previous_val"] = self.diz_all[w][i_ese][i_column]  # salviamo il testo che c'era prima, in questo modo in caso di errori possiamo
                # facilmente rimetterlo usando questa stessa funzione

        self.diz_all[w][i_ese][i_column] = text

        # ricostruzione timer msg, usa esc_dizAll
        counter_massimo = self.counter_dict[w][i_ese]
        counter_minimo = self.counter_dict[w][i_ese-1] if i_ese != 0 else 0  # è 0 se è il primo esercizio
            # visto che counter 0 = primo esercizio prima serie, altrimenti è il limite massimo dell'es prima
        for i_serie, counter in enumerate(range(counter_minimo, counter_massimo)):  # counter: counter del timer_msg da mettere nella flat list, i_serie: serve per fare N serie su N
            self.flat_timer_msg_list[w][counter] = eseSerie_to_timerMsg(self.diz_all, w, i_ese, i_serie)


    def compile_lowLevel(self, text, i_ese, dizAll_idx, uid, forced_reps=False, change_buttons=None, bot_probs=None, callback=False,
                         CB_new_dizAll_idx=None, CB_new_val=None):
        """la funzione che viene usata quando si sta compilando una nuova scheda e vengono inseriti i valori"""
        w = self.usermode

        mk = ">" if not callback or CB_new_dizAll_idx else "╳"  # simbolo usato
        type = DIZALL_IDX_TO_TYPE[dizAll_idx]

        if dizAll_idx == 3:  # serie
            self.diz_all[w][i_ese][3] = int(text)
            self.adapt_to_newSerie(w)

        elif dizAll_idx == 0:  # pausa
            self.apply_modification(int(text[:-1]), i_ese, 0, True)
            self.flat_pause_list[w] = [self.diz_all[w][ese][0] for ese in range(len(self.serie_in_ese_list[w]))
                                        for serie in range(self.diz_all[w][ese][3])]

        elif forced_reps:  # caso in cui è stato scritto "N reps"
            self.apply_modification(text[:-5], i_ese, 2, True)

        elif dizAll_idx in [1, 2, 4, 5]:
            # applicazione e messaggio
            self.apply_modification(text, i_ese, dizAll_idx, True)
            if CB_new_dizAll_idx:
                self.apply_modification(CB_new_val, i_ese, CB_new_dizAll_idx, True)

        txt_scheda, _ = self.invia_messaggioScheda(send=False)
        try:
            bot.edit_message_text(text=txt_scheda, chat_id=uid, message_id=g_dict["scheda_msgId"], parse_mode="HTML")
        except:
            pass

        if not callback and not CB_new_dizAll_idx:
            text = f"{self.diz_all[w][i_ese][dizAll_idx]}<code> {mk} </code><b>{type}</b>"
        else:
            if CB_new_dizAll_idx:
                type = DIZALL_IDX_TO_TYPE[CB_new_dizAll_idx]
            text = f"{CB_new_val}<code> {mk} </code><b>{type}</b>"
        if dizAll_idx in [1, 2, 4, 5] and not callback:
            text += f"\n🤖:<code> {bot_probs}</code>"

        msg_ = bot_trySend(text, change_buttons)
        return msg_

    def compile(self, w, msg):
        i_ese = self.current_ese_idx

        # qualsiasi modifica facciamo a questo step trasforma l'esercizio in un esercizio reale, facendo passare note da False a None
        self.diz_all[w][i_ese][5] = None
        bot_tryDelete(A_UID, msg.id)

        # SERIE
        if msg.text in STR_1_6_LIST:  # format corretto: N
            self.compile_lowLevel(msg.text, i_ese, 3, A_UID)

        # PAUSA
        elif msg.text[-1] == "\"" and is_digit(msg.text[:-1]):  # format corretto: NN"
            self.compile_lowLevel(msg.text, i_ese, 0, A_UID)

        # CASO "N REPS"
        elif msg.text.endswith("reps") and len(msg.text) < 7:
            self.compile_lowLevel(msg.text, i_ese, 2, A_UID, forced_reps=True)

        # NOME, REPS, PESO, NOTE
        else:
            # pred
            model_input = tf.expand_dims(tf.constant(msg.text), 0)
            pred = g_dict["msgLabelling_model"](model_input)
            pred_idx = int(tf.argmax(pred, 1))
            type = PRED_IDX_TO_TYPE[pred_idx]  # Nome, Reps, ...

            # testo da aggiungere al messaggio
            probs_dict = {key: round(float(pred[0][i]), 2) for key, i in zip(["E", "R", "P", "N"], range(4))}
            s_probs_dict = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
            probs_text = [f"{key} {prob:.0%}" for key, prob in s_probs_dict.items()]
            probs_text = "  ".join(probs_text)

            # bottoni per cambiare tipo
            ic([LETTER_TO_TYPE[key] for key in s_probs_dict][1:])
            ic(s_probs_dict)
            texts_list = [LETTER_TO_TYPE[key] for key in s_probs_dict][1:] + ["✖️ Undo"]
            change_buttons_raw = dict(texts=texts_list, callbacks=[f"CAMB{pred_idx}{i}" for i in texts_list])  # il callback è composto dal prev_idx del tipo
                # previsto e dalla stringa del tipo selezionato per cambiare
            change_buttons = inline_buttons_creator(change_buttons_raw, 4)
            self.creazioneScheda_undo["cambioScheda_msg"] = msg.text

            # rimozione buttons a scorso messaggio di predizione tipo
            if self.creazioneScheda_undo["cambioScheda_msgId"] != None:  # solo se sono già stati mandati messaggi di compilazione
                try:  # try block perchè potrebbe essere un undo e quindi il messaggio è stato eliminato
                    bot.edit_message_reply_markup(A_UID, self.creazioneScheda_undo["cambioScheda_msgId"], reply_markup=None)
                except:
                    pass

            msg_ = self.compile_lowLevel(msg.text, i_ese, TYPE_TO_DIZALL_IDX[type], A_UID, change_buttons=change_buttons, bot_probs=probs_text)
            # noinspection PyTypeChecker
            self.creazioneScheda_undo["cambioScheda_msgId"] = msg_.message_id



T = Training()


# THREAD MESSAGGI RICEVUTI

def thread_telegram():
    def listener(messages):
        for msg in messages:
            uid = msg.chat.id

            if uid == A_UID:
                msg_id = msg.message_id
                w = T.usermode

                if msg.text == "/test":
                    print("test")
                    ic(msg_id)
                    ic(msg)

                    keyboard_workous = ReplyKeyboardMarkup()
                    keyboard_workous.row(KeyboardButton("Addominali"))
                    keyboard_workous.row(KeyboardButton("Petto Bi"), KeyboardButton("Schiena"))
                    # buttons = [InlineKeyboardButton("ciao")]/
                    # keyboard = InlineKeyboardMarkup(row_width=1)

                    a = bot.send_message(uid, "OK", reply_markup=keyboard_workous)

            # """COMANDI GENERALI"""

                # /START
                elif msg.text == ALL_COMMANDS_DICT["/start"]:
                    T.select_workout(msg_id)

                # /COMANDI
                elif msg.text == ALL_COMMANDS_DICT["/comandi"]:
                    msg = "<b>Comandi generali</b>"
                    for comando, desc in DICT_COMANDI_GEN.items():
                        msg += f"\n<code>{comando}<code> - {desc}"
                    msg += "\n<b>Comandi durante allenamento</b>"
                    for comando, desc in DICT_COMANDI_DUR.items():
                        msg += f"\n<code>{comando}<code> - {desc}"

                    bot_trySend(msg)

                # /SINTASSI
                elif msg.text == ALL_COMMANDS_DICT["/sintassi"]:
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
                    bot.send_message(uid, msg_sintassi, parse_mode=MD)

                # /DATABASE
                elif msg.text == ALL_COMMANDS_DICT["/database"]:
                    send_db_json()
                    bot.send_message(uid, "per informazioni su come dovrebbe essere il database usare /info")

                # /INFO
                elif msg.text == ALL_COMMANDS_DICT["/info"]:
                    info = """
                    <b>⚠️ Informazioni utili ⚠️</b>
                    Database
                    •  <code>data_prossima_stat</code> dovrebbe essere di domenica e a 4 settimane dall'ultima statistica di 4 settimane
                    •  <code>calendario</code> dovrebbe essere il mese in cui siamo in questo momento
                    •  <code>data_prossima_stat_m</code> dovrebbe essere l'ultimo giorno di questo mese
                    •  <code>Cycle_dailyLever</code> dovrebbe essere True se l'orario è prima delle 22:00, altrimenti False
                    •  <code>Cycle_4settLever</code> e <code>Cycle_monthLever</code> devono essere True se non è stato inviato un grafico entro 2 ore fa
                    
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
                elif msg.text == ALL_COMMANDS_DICT["/nuovascheda"]:
                    cambiaScheda_button = dict(texts=["Cambia scheda"], callbacks=["NUOV"])
                    button = inline_buttons_creator(cambiaScheda_button, 1)

                    g_dict["nuovaScheda_msgId"] = msg_id
                    T.callback_antiSpam = False
                    bot.send_message(msg.chat.id, text="Sei sicuro di voler cambiare scheda? Usare questo comando solo se si cambia la scheda, se lo hai già fatto per questa "
                                                       "scheda non rifarlo, ti basterà selezionare gli esercizi da compilare con /start",
                                     reply_markup=button)


                # 1 /ARCHIVIO
                elif msg.text == ALL_COMMANDS_DICT["/archivio"]:
                    date_schede = list(T.startup_db["schede"].keys())
                    date_schede.sort()
                    g_dict["date_schede"] = date_schede
                    raw_buttons = {"texts": [], "callbacks": []}

                    for scheda in date_schede:
                        anno_inizio, sett, data_inizio, data_fine, index, ultima_scheda = schedaFname_to_title(scheda, date_schede)

                        if not ultima_scheda:
                            raw_buttons["texts"].append(f"{anno_inizio}:  {weekday_nday_month_year(data_inizio)} - {weekday_nday_month_year(data_fine)}, {sett} sett")
                        else:
                            raw_buttons["texts"].append(f"Attuale: {weekday_nday_month_year(data_inizio)}, {sett} sett")

                        raw_buttons["callbacks"].append(f"ARCH{index}")

                    buttons = inline_buttons_creator(raw_buttons, row_width=1)

                    T.callback_antiSpam = False
                    bot.send_message(uid, "Seleziona quale scheda visualizzare", reply_markup=buttons)


                # /TEMPI
                elif msg.text == ALL_COMMANDS_DICT["/tempi"]:
                    g_dict["inizioAll_HH_MM_SS"] = ora_EU(soloOre=True)
                    msg = f"""<b>MEDIA TEMPO ULTIMI 3 WORKOUT:</b>\n"""
                    for w in T.active_workouts:
                        msg += f"<b>{NOMI_IN_SCHEDA[w]}</b>: {T.workout_finish_secs(w, False)}, {secs_to_hours(sum(T.avg_adj_prev_tempo[w]))}"

                    bot_trySend(msg)

                # 1 /SCORE
                elif msg.text == ALL_COMMANDS_DICT["/score"]:
                    bot.send_message(msg.chat.id, text="Seleziona l\'esercizio che vuoi segnare", reply_markup=KEYBOARD_WORKOUTS)
                    T.sequences_usermode = "score"

                # 2
                elif T.sequences_usermode == "score":
                    if msg.text in NOMI_WORKOUT:
                        T.end_workout(score_w=START_DICT[msg.text])
                    else:
                        bot.send_message(msg.chat.id, text=f"Seleziona un esercizio valido", reply_markup=KEYBOARD_WORKOUTS)


                # /MSGDEBUG
                elif msg.text == ALL_COMMANDS_DICT["/msgdebug"]:
                    pass

                # /EXIT
                elif msg.text == ALL_COMMANDS_DICT["/exit"]:
                    bot_tryDelete(uid, msg_id)
                    bot_tryDelete(uid, msg_id+1)

                    T.sequences_usermode = "none"

                # /FORCE TRAINING
                elif msg.text == "/forcetraining":
                    start_training_procedure()

                # 1 /PESO
                elif msg.text == ALL_COMMANDS_DICT["/peso"]:
                    bot.send_message(msg.chat.id, text=f"Inviare il numero di mesi precedenti da cui prendere i dati")
                    T.sequences_usermode = "grafico_peso"

                # 2     /PESO
                elif T.sequences_usermode == "grafico_peso":
                    try:
                        num_months = int(msg.text)
                        g_dict["num_months"] = num_months
                        T.send_grafico_peso = True

                    except:
                        bot.send_message(msg.chat.id, text=f"Inviare un numero valido")



            # DURANTE ESERCIZIO

                elif T.usermode in LISTA_ES_SIMPLE:

                    # /DONE
                    if msg.text == ALL_COMMANDS_DICT["/done"]:
                        if not T.run_timer:  # se il timer non sta già andando
                            T.done(msg)
                        else:
                            bot.delete_message(uid, msg_id)

                    # /NOTIME
                    elif msg.text == ALL_COMMANDS_DICT["/notime"]:
                        T.notime = True
                        bot.send_message(uid, "Il tempo che verrà impiegato per finire questo allenamento non verrà salvato")

                    # SKIP /SK
                    elif msg.text == ALL_COMMANDS_DICT["/sk"]:
                        T.skip = True
                        bot.delete_message(chat_id=uid, message_id=msg_id)

                    # /BACK
                    elif msg.text == ALL_COMMANDS_DICT["/back"]:
                        # todo non è finito
                        T.counter -= 1
                        bot.delete_message(chat_id=uid, message_id=msg_id)

                    # 1 SWITCH /SW
                    elif msg.text == ALL_COMMANDS_DICT["/sw"]:
                        bot.delete_message(chat_id=uid, message_id=msg_id)

                        keyb_switch = ReplyKeyboardMarkup(one_time_keyboard=True)
                        for ese in range(len(T.switch_list[w])):
                            keyb_switch.add(T.switch_list[w][ese])
                        msg_switch = bot.send_message(msg.chat.id, text=f"Scegli a quale esercizio andare", reply_markup=keyb_switch)

                        g_dict["switch_msgId"] = msg_switch.message_id
                        T.sequences_usermode = "switch"

                    # 2     SECONDO PASSO SWITCH
                    elif T.sequences_usermode == "switch":
                        T.switch_ese(msg.text, msg_id)

                    # MODIFICHE ALLA SCHEDA

                    # 1, /comando
                    elif msg.text in ["/p", "/ese", "/n", "/rep"]:
                        bot_tryDelete(uid, msg_id)

                        if MODIFICHE_DICT[msg.text]["db_idx"] == 4:
                            keyboard = ReplyKeyboardMarkup()
                            keyboard.add(*T.peso_ese_list[w])
                        else:
                            keyboard = T.ese_names_keyb[w]

                        msg_ = bot.send_message(uid, text=MODIFICHE_DICT[msg.text]["msg0"], reply_markup=keyboard, parse_mode=MD)
                        g_dict["msg_mod"] = msg_.message_id
                        T.sequences_usermode = msg.text + "1"  # /ese1

                    # 2, alla selezione del nuovo esercizio
                    elif T.sequences_usermode in ["/p1", "/ese1", "/n1", "/rep1"]:
                        bot_tryDelete(uid, msg_id)
                        bot_tryDelete(uid, g_dict["msg_mod"])

                        tipo_modifica = T.sequences_usermode[:-1]  # es: /ese
                        iteration_list = T.peso_ese_list if tipo_modifica == "/p" else T.switch_list_full

                        for ese in range(len(iteration_list[w])):
                            if msg.text == iteration_list[w][ese]:
                                nome_ese = T.switch_list_full[w][ese]
                                msg_ese = f"Esercizio selezionato: <b>{nome_ese}</b>"

                                if T.sequences_usermode == "/p1":
                                    msg_ = bot_trySend(f"{msg_ese}\n<code>{T.diz_all[w][ese][4].replace('㎏', 'k')}</code>, scrivere il nuovo peso. \n"
                                                       f"Scrivere il peso seguendo le regole sulla scrittura del peso che si trovano su /info")
                                elif T.sequences_usermode == "/ese1":
                                    msg_ = bot_trySend(f"Esercizio selezionato: <code>{nome_ese}</code>, scrivere il nuovo nome")
                                elif T.sequences_usermode == "/rep1":
                                    msg_ = bot_trySend(f"{msg_ese} <code>{T.diz_all[w][ese][2]}</code> scrivere le nuove reps")
                                elif T.sequences_usermode == "/n1":
                                    msg_ = bot_trySend(f"{msg_ese} <code>{T.diz_all[w][ese][5]}</code>, aggiungere le note")

                                g_dict["msg_mod"] = msg_.message_id
                                g_dict["idx_ese_cambiato"] = ese
                                T.sequences_usermode = tipo_modifica + "2"  # /ese2

                                break

                    # 3, alla scrittura del nuovo valore
                    elif T.sequences_usermode in ["/p2", "/ese2", "/n2", "/rep2"]:
                        bot_tryDelete(uid, msg_id)
                        bot_tryDelete(uid, g_dict["msg_mod"])

                        tipo_modifica = T.sequences_usermode[:-1]

                        ese = g_dict['idx_ese_cambiato']
                        col_idx = MODIFICHE_DICT[tipo_modifica]["db_idx"]

                        previous_val = copy.deepcopy(T.diz_all[w][ese][col_idx])

                        # scrittura in db e aggiornamento diz all
                        T.apply_modification(msg.text, ese, col_idx)

                        nome_ese = T.diz_all[w][ese][1]
                        nome_ese_str = f"<b>{nome_ese}</b>:\n"

                        # /p
                        if T.sequences_usermode == "/p2":
                            peso_ese = T.diz_all[w][ese][4]
                            db_insert("aumenti_peso", [ora_EU(isoformat=True),
                                                       previous_val,
                                                       msg.text,
                                                       nome_ese],
                                      mode="append")

                            # messaggio aumento / diminuizione peso
                            if stringaPeso_to_num(previous_val) < stringaPeso_to_num(msg.text):
                                reply_msg = f"{nome_ese_str}💹 {previous_val} <b>→</b> {peso_ese}"
                                db_insert(["aumenti_msgScheda", w], {"w_count": db["workout_count"][w],
                                                                     "ese_idx": ese,
                                                                     "peso": previous_val},
                                          mode="append")
                            else:
                                reply_msg = f"{nome_ese_str}🈹 {previous_val} <b>→</b> {peso_ese}"

                        # /ese
                        elif T.sequences_usermode == "/ese2":
                            reply_msg = f"<i>{previous_val}</i> <b>→</b> <i>{nome_ese}</i>"

                        # /rep
                        elif T.sequences_usermode == "/rep2":
                            rep = T.diz_all[w][ese][2]
                            reply_msg = f"{nome_ese}<i>{previous_val}</i> <b>→</b> <i>{rep}</i>"

                        # /n
                        elif T.sequences_usermode == "/n2":
                            note = T.diz_all[w][ese][5]
                            # se c'erano già delle note per questo esercizio
                            if previous_val is not None:
                                reply_msg = f"{nome_ese_str}<i>{previous_val}</i> <b>→</b> <i>{note}</i>"
                            else:
                                reply_msg = f"{nome_ese_str}note aggiunte: <i>{note}</i>"

                        msg_ = bot_trySend(reply_msg, reply_markup=KEYBOARD_DONE)
                        preserve_msg(msg_)

                        # nuovi dati per il modello
                        db_insert("new_training_data", [msg.text, MODIFICHE_DICT[tipo_modifica]["label"]], mode="append")

                        T.sequences_usermode = "none"

                    # CREAZIONE SCHEDA

                    # /FINE
                    elif msg.text == ALL_COMMANDS_DICT["/fine"]:
                        T.fine_creazione_scheda()

                    # COMPILAZIONE SCHEDA
                    elif T.creazione_scheda and msg.text[0] != "/":
                        T.compile(w, msg)

                    else:
                        bot.send_message(msg.chat.id, text=f"Comando sconosciuto")


                # non durante allenamento ma usa comando dove è richiesto essere in allenamento
                elif msg.text in DICT_COMANDI_DUR:
                    bot.send_message(msg.chat.id, text=f"Per usare {msg.text} devi prima selezionare un allenamento con /start")


                # SCORE PESO, FUORI DAL TRAINING
                elif msg.text[2] in [".", ","] and len(msg.text) == 5:
                    for i in [0, 1, 3, 4]:  # index dei numeri in msg.text
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
                        bot_trySend("Il format da seguire per registrare il peso è: <code>NN.NN</code> o <code>NN,NN</code>")

                # ERRORE
                else:
                    bot.send_message(msg.chat.id, text="Comando sconosciuto")


    ##################


    # CALLBACK HANDLER

    @bot.callback_query_handler(func=lambda call: True)
    def callback(call):
        uid = call.message.chat.id
        # msg_id = call.message.id
        code = call.data[0:4]
        after_code = call.data[4::]


        # CAMBIO TIPO IN CREAZIONE SCHEDA e /UNDO
        if code == "CAMB":
            # es call.data: CAMB0Reps  da 0 (nome) si vuole passare a reps
            previous_dizAll_idx = int(after_code[0])  # idx dell'esercizio che vogliamo cambiare
            new_dizAll_idx = TYPE_TO_DIZALL_IDX[after_code[1:]] if after_code[1:] in list(TYPE_TO_DIZALL_IDX.keys()) else None  # idx del nuovo esercizio

            bot_tryDelete(uid, T.creazioneScheda_undo["cambioScheda_msgId"])
            # noinspection PyTypeChecker
            T.compile_lowLevel(T.creazioneScheda_undo["previous_val"], T.current_ese_idx, previous_dizAll_idx, uid,
                               callback=True, CB_new_dizAll_idx=new_dizAll_idx, CB_new_val=T.creazioneScheda_undo["cambioScheda_msg"])

        # questi callback hanno bisogno della protezione antispam perchè i messaggi non vengono eliminati
        if not T.callback_antiSpam:
            T.callback_antiSpam = True

            # CONFERMA DI /NUOVASCHEDA
            # todo aggiustare comilazione nuovascheda
            if code == "NUOV":
                # creazione struttura nuova scheda nel db
                YYYY_MM_DD = str(ora_EU())[:10]
                db_insert(["schede", YYYY_MM_DD],
                          dict(addominali=STRUTTURABASE_DIZALL, pettobi=STRUTTURABASE_DIZALL, schiena=STRUTTURABASE_DIZALL,
                               spalletri=STRUTTURABASE_DIZALL, altro=STRUTTURABASE_DIZALL))

                # reset w sec taken
                for w in LISTA_ES_SIMPLE:
                    db_insert(["w_sec_taken", w], [])

                # GRAFICO RADAR SCHEDA PRECEDENTE
                T.send_radar_graph = True
                time.sleep(1)

                # db data cambioscheda
                db_insert("data_cambioscheda", ora_EU(isoformat=True), mode="append")

                msg_ = bot.send_message(uid, text="La scheda è stata resettata, riavvio del bot in corso...")
                T.restart_msgId = msg_.message_id
                T.restart = True
                T.select_workout(g_dict["nuovaScheda_msgId"])

            # INIZIO WORKOUT e NUOVA SCHEDA
            if code == "WORK":
                T.start_workout(w=after_code)

            # SCHEDA ARCHIVIO
            elif code == "ARCH":
                index = int(after_code)
                scheda_YYYY_MM_DD = g_dict["date_schede"][index]
                img_path = f"schede/immagini/{scheda_YYYY_MM_DD[:-5]}.jpeg"

                if os.path.isfile(img_path) and index != len(g_dict["date_schede"]) - 1:  # se c'è già l'immagine e non è la scheda attuale
                    loaded_img = open(img_path, "rb")
                    bot.send_document(uid, loaded_img)

                else:
                    msg_clessidra = bot.send_message(uid, text="⏳")

                    loaded_img = schedaIndex_to_image(T.startup_db["schede"][scheda_YYYY_MM_DD], scheda_YYYY_MM_DD)

                    bot.delete_message(uid, msg_clessidra.message_id)
                    msg_ = bot.send_document(uid, loaded_img)
                    preserve_msg(msg_)


    bot.set_update_listener(listener)
    while True:
        # bot.polling(none_stop=True)  # per debug: uncommentare questa linea
        try:
            bot.polling(none_stop=True)
        except Exception as exc:
            traceback_message(exc)


t = threading.Thread(target=thread_telegram)
t.start()

addTo_botStarted(f"Functions, packages and variables loaded, message handling ready")

#######################


# CYCLE (MAIN THREAD)

while True:
    cycle_seconds = 600
    for _ in range(cycle_seconds):
        if T.send_grafico_peso:
            try:
                fname, msg = Grafico_peso(db["peso_bilancia"], g_dict["num_months"])
                img = open(fname, "rb")
                msg_ = bot.send_photo(A_UID, img, caption=msg)
                preserve_msg(msg_)
                T.sequences_usermode = "none"

            except Exception as exc:
                bot.send_message(A_UID, f"C'è stato un errore ({exc})")
                split_msg(A_UID, f"TRACEBACK:\n\n{traceback.format_exc()}")

            T.send_grafico_peso = False

        elif T.send_radar_graph:
            try:
                img_grafico_WS = Grafico_radarScheda(db["dizAll_2volte"], T.diz_all, db["data_2volte"], db["workout_count"])
                bot.send_photo(A_UID, img_grafico_WS, caption=f"<b>Radar graph</b> <code>{weekday_nday_month_year(ora_EU(isoformat=True), year=True)}</code>", parse_mode="HTML")

            except Exception as exc:
                bot.send_message(A_UID, text="Il radar graph ha fallito:")
                traceback_message(exc)

            T.send_radar_graph = False

        time.sleep(1)

    n_ora = numero_ora()
    sec_22 = 60*60*22

    # # se sono passate le 22:00
    # if 1 == 1:
    if n_ora > sec_22 and db["Cycle_processLever"]:
        data_oggi = ora_EU()
        data_oggi_iso = ora_EU(isoformat=True)

        if db["Cycle_dailyLever"]:
            # RIMOZIONE MESSAGGI
            clear_msg = bot.send_message(A_UID, "Pulizia dei messaggi inutili in corso...")
            l_msgId = db["latest_msgId"]

            if actual_environment:  # solo se è da replit e non da pycharm, in questo modo non vengono eliminati tutti i messaggi (visto che il db non è quasi mai aggiornato)
                for msg_id in range(l_msgId + 1, clear_msg.message_id + 1):
                    bot_tryDelete(A_UID, msg_id)
                pass
                db_insert("latest_msgId", clear_msg.message_id)

            # TRAINING MODELLO
            if len(db["new_training_data"]) > 50:
                start_training_procedure()

            db_insert("Cycle_dailyLever", False)


        # GRAFICO 4 SETTIMANE

        data_prossima_stat = dt.datetime.fromisoformat(db["data_prossima_stat"])

        # se sono passati 28 giorni dall'ultima volta che ci sono state le statistiche, oppure: se la data prossima stat nel database è prima di quella di oggi
        # sarebbe oggi ma per sicurezza mettiamo la data che dovrebbe essere per evitare che il grafico venga sballato in caso il processo non venga eseguito nel giorno giusto
        # if 1 == 1:
        if data_prossima_stat < data_oggi and db["Cycle_4settLever"]:

            try:
                bot.send_message(A_UID, "GRAFICO 4 SETTIMANE:")

                msg, img, sum_y_allenamenti, sum_y_addominali, sum_y_aumenti_peso = Grafico_4sett(db, data_prossima_stat, somma_numeri_in_stringa, M_TO_MESI)
                msg_ = bot.send_photo(A_UID, img, caption=msg)
                preserve_msg(msg_)

                # DB
                # settiamo la data della prossima data nel databse
                db_insert("data_prossima_stat", (data_prossima_stat + dt.timedelta(days=28)).isoformat())

                # salviamo nel database i dati per il prossimo grafico mensile
                db_insert(["ultima_stat", "y_allenamenti"], sum_y_allenamenti)
                db_insert(["ultima_stat", "y_addominali"], sum_y_allenamenti)
                db_insert(["ultima_stat", "y_aumenti_peso"], sum_y_aumenti_peso)

                db_insert("Cycle_4settLever", False)

            except Exception as exc:
                msg_ = bot.send_message(A_UID, f"C'è stato un errore nel grafico delle 4 settimane")
                preserve_msg(msg_)
                msg_ = traceback_message(exc)
                preserve_msg(msg_)


        # ITERAZIONE MENSILE

        # se i giorni tra oggi e la data in cui ci sarebbe il prossimo grafico sono meno di 1
        # if 1 == 1:
        if (dt.datetime(db["data_prossima_stat_m"][0], db["data_prossima_stat_m"][1], db["data_prossima_stat_m"][2]) - data_oggi).days < 1 and db["Cycle_monthLever"]:

            try:
                # GRAFICO DEL PESO
                bot.send_message(A_UID, "GRAFICO PESO:")
                months = 3
                res = Grafico_peso(db["peso_bilancia"], months)
                if res:
                    fname, msg = res
                    img = open(fname, "rb")
                    msg_ = bot.send_photo(A_UID, img, caption=msg)
                    preserve_msg(msg_)

                else:
                    bot.send_message(A_UID, f"Nessun pesamento negli ultimi {months} mesi")

                # BACKUP DATABASE
                bot.send_message(A_UID, "BACKUP DATABASE")
                send_db_json()

                # GRAFICO ALLENAMENTI MENSILE
                bot.send_message(A_UID, "GRAFICO MENSILE:")

                msg, img = Grafico_mensile(db, somma_numeri_in_stringa, M_TO_MESI, ora_EU, actual_environment)
                msg_ = bot.send_photo(A_UID, img)
                preserve_msg(msg_)
                msg_ = bot_trySend(msg)
                preserve_msg(msg_)

                # DB
                calendario_list = db["calendario"]
                calendario_list[1] += 1
                if calendario_list[1] == 13:
                    calendario_list[1] = 1
                    calendario_list[0] += 1

                db_insert("data_prossima_stat_m", [calendario_list[0], calendario_list[1], calendar.monthrange(calendario_list[0], calendario_list[1])[1]])
                db_insert("Cycle_monthLever", False)

            except Exception as exc:
                msg_ = bot.send_message(A_UID, f"C'è stato un errore nell'iterazione mensile")
                preserve_msg(msg_)
                msg_ = traceback_message(exc)
                preserve_msg(msg_)

        db_insert("Cycle_processLever", False)  # false vuol dire che il processo è finito
        # reboot alla fine
        # if not actual_environment:
        #     with open("temporanei/db.json", "w") as file:
        #         json.dump(db, file, indent=4)


    # RESET LEVERS A MEZZANOTTE
    elif 0 < n_ora < sec_22:
        db["Cycle_processLever"] = True
        db["Cycle_monthLever"] = True
        db["Cycle_4settLever"] = True
        db["Cycle_dailyLever"] = True
