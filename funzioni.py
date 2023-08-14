""" questo file contiene funzioni usate sia da main.py che da creazione_immagini.py """
# import os
import datetime as dt
import pytz


from icecream import ic
wrap_width = 195
def to_string(obj):
    """ obj: contenuto di quello che viene printato, es: ic("aaa"), "aaa" è obj, ma non è per forza una stringa, può essere anhce un numero, lista, ecc. """
    def element_wrap(wrap_width, text, key=None):
        # se ha una key viene trattato come un dizionario
        string_split = text.split(" ")
        str_split_lengths = [len(i) + 1 for i in string_split]
        newline_idxs = []
        new_str = ""
        newline_i = 0
        n_line_padding = 0
        last_split = 0

        for i_split, split_ in enumerate(string_split):
            if sum(str_split_lengths[i] for i in range(last_split, i_split + 1)) + n_line_padding > wrap_width:
                last_split = i_split
                newline_idxs.append(last_split)
                n_line_padding = 1

        more_newlines = True if len(newline_idxs) > 0 else False

        for i, split_ in enumerate(string_split):
            if more_newlines and i == newline_idxs[newline_i]:
                if not key:  # non dizionario
                    new_str += "\n" + " " + split_ + " "
                else:  # dizionario
                    new_str += "\n" + " " * (len(key) + 6) + split_ + " "

                newline_i += 1
                if newline_i >= len(newline_idxs):
                    more_newlines = False

            else:
                new_str += split_ + " "

        return new_str

    if type(obj) == dict:
        new_str = "{"
        for key, value in obj.items():
            dict_wrap_width = wrap_width - (len(str(key)) + 5)
            new_value = element_wrap(dict_wrap_width, str(value), key=key)

            q = '"' if type(key) == str else ''
            new_str += f'{q}{key}{q}: {new_value[:-1]},\n '
        new_str = new_str[:-3] + "}"

    else:
        new_str = element_wrap(wrap_width, str(obj))

    return new_str
ic.configureOutput(prefix="> ", includeContext=True, argToStringFunction=to_string)




def md_v2_replace(string, exceptions=[], reverse=False):
    """ funzione che fa l'escape ai caratteri del markdown, exceptions serve per togliere determinati caratteri dall'escaping e reverse serve per portare da escaped a non
    escaped """
    if type(string) == str:
        replacements = ["+", "-", "(", ")", "|", ".", "="]
        [replacements.remove(excep) for excep in exceptions]

        replacements = [(char, f"\\{char}") if reverse == False else (f"\\{char}", char) for char in replacements]

        for rep_this, with_this in replacements:
            string = string.replace(rep_this, with_this)
        return string
    else:
        return string

def is_digit(string):
    try:
        float(string)
        return True
    except:
        return False


def ora_EU(tzinfo=None, isoformat=False, soloOre=False):
    """ funzione che returna l'ora di Roma e che di default toglie le informazione di timezone, in questo modo possiamo sottrarre date che non hanno timezone a questa data """
    if tzinfo:
        data = dt.datetime.now(pytz.timezone("Europe/Rome"))
    else:
        data_oggi = dt.datetime.now(pytz.timezone("Europe/Rome"))
        data = data_oggi.replace(tzinfo=None)

    if isoformat or soloOre:
        data = data.isoformat()
        if soloOre:
            data = data[11:19]

    return data


def secs_to_hours(sec, show_seconds=False):
    """ es: secs_to_hours(11545) = 3h 12m 25s, secs_to_hours(600) = 10m 0s"""
    HH_MM_SS = str(dt.timedelta(seconds=sec))
    HMS_split = HH_MM_SS.split(":")

    # round dei minuti
    if not show_seconds:
        if float(HMS_split[2]) > 30:
            HMS_split[1] = str(int(float(HMS_split[1])) + 1)

    seconds = str(int(float(HMS_split[2]))) + "s" if show_seconds else ""
    mins = str(int(float(HMS_split[1]))) + "m "
    ore = str(int(float(HMS_split[0]))) + "h " if HMS_split[0] != "0" else ""
    # str(int()) serve per togliere gli 0, es: 08 -> 8

    return ore + mins + seconds


def weekday_nday_month_year(time, weekday=False, year=False):
    """ time -> mar 8 feb 2023 """
    m_to_mesi = {"01": "Gen", "02": "Feb", "03": "Mar", "04": "Apr", "05": "Mag", "06": "Giu", "07": "Lug", "08": "Ago", "09": "Set", "10": "Ott", "11": "Nov", "12": "Dic"}

    time = time.isoformat() if type(time) != str else time
    (y, month, daynum) = time[0:10].split("-")
    weekday_int = dt.datetime.fromisoformat(time).weekday()  # 0 = lunedì, 6 = domenica
    giorni_sett = {0: "lun ", 1: "mar ", 2: "mer ", 3: "gio ", 4: "ven ", 5: "sab ", 6: "dom "}

    Y = ""
    WD = ""

    if weekday:
        WD = giorni_sett[weekday_int]
    if year:
        Y = " " + y

    return f"{WD}{daynum} {m_to_mesi[month]}{Y}"

def schedaFname_to_title(YYYY_MM_DD, schede_list):
    ultima_scheda = False
    Y, M, D = YYYY_MM_DD.split("-")
    data_inizio = dt.datetime(int(Y), int(M), int(D))

    i = schede_list.index(YYYY_MM_DD)  # rappresenta il numero della scheda nella lista formata da tutte le schede sorted

    if i != len(schede_list)-1:  # se non è l'ultima scheda
        Y_ini, M_ini, D_ini = schede_list[i+1]
        data_fine = dt.datetime(int(Y_ini), int(M_ini), int(D_ini))
    else:
        ultima_scheda = True
        data_fine = ora_EU()

    anno_inizio = data_inizio.year
    sett = round((data_fine - data_inizio).days / 7)

    return anno_inizio, sett, data_inizio, data_fine, i, ultima_scheda


def stringaPeso_to_num(stringa):
    # todo testare più a fondo
    """
    trasforma le stringhe di peso in numeri di kg utili per il radar graph trasformando i dischi extra in 1.5 e facendo la media tra i kg
    es:
    "36k + 1 (- 1), 27,5k + 2"  ->  34.0        "10k + 3"   ->  14.5
    "36k + 1 (- 1)"             ->  37.5        "3"         ->  4.5
    "36k + 1"                   ->  37.5        "abcde"     ->  0
    "20k | 10k"                 ->  15.0
    """
    peso_disco_macchinario = 1.5  # rappresenta il peso di un disco extra di un macchinario in kg

    stringa = stringa.replace(",", ".")  # sostituiamo le , coi . in modo da poter usare float()
    stringa = stringa.replace("㎏", "k")
    split = stringa.split()

    # prima vengono selezionati solo i pieces che non contengono parentesi e poi quelli che contengono numeri
    keep_idxs = []
    for piece_i, piece in enumerate(split):
        if "(" not in piece and ")" not in piece:
            if any(char.isdigit() for char in piece):
                keep_idxs.append(piece_i)

    split = [split[idx] for idx in keep_idxs]

    # se ha k l'idx viene aggiunto a k_idxs, altrimenti a num idxs
    k_idxs = []
    num_idxs = []
    [k_idxs.append(idx) if "k" in piece else num_idxs.append(idx) for idx, piece in enumerate(split)]

    division_factor = max(1, len(k_idxs))  # numero per cui dividiamo la somma finale per ottenere la media, se non ci sono k_idxs è 1

    for num_idx in num_idxs:
        split[num_idx] = float(split[num_idx]) * peso_disco_macchinario

    for k_idx in k_idxs:
        split[k_idx] = float(split[k_idx].split("k")[0])

    average = round(sum(split) / division_factor, 2)
    return average

