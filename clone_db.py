import copy
import replit
import json

def clone_database(db):
    """Nota importante:  qualsiasi cosa nel database che sia uguale a "])" verrà convertita a "]", quindi è sconsigliato l'utilizzo di sets e questo aspetto va preso in
    considerazione per le stringhe che si aggiungono nel database. Inoltre gli unici 2 tipi di dato (oltre a quelli classici) compatibili sono i dizionari e le liste,
    quindi fare attenzione prima di usare nuovi tipi perchè potrebbero creare errori nella clonazione
    Inoltre True, False, None all'interno delle stringhe potrebbero essere convertiti rispettivamente in true, false, null"""

    # creazione del primo database che toglie alcune observed... usando .value
    raw_db = copy.deepcopy(db)
    cloned_db = {}

    for key, val in raw_db.items():
        if type(val) in [replit.database.database.ObservedList, replit.database.database.ObservedDict]:
            cloned_db[key] = val.value
        else:
            cloned_db[key] = val

    raw_db_string = "{"
    for i, (key, val) in enumerate(cloned_db.items()):
        q = "'" if type(val) == str else ""
        raw_db_string += (f"'{key}': {q}{val}{q}")
        raw_db_string += ", " if i != len(cloned_db)-1 else ""
    raw_db_string += "}"

    with open('raw_db.txt', 'w') as f:
        f.write(raw_db_string)

    # scrittura del dizionario su file di testo
    with open('raw_db.txt') as f:
        raw_db_string = f.readlines()

    # replacements iniziali per essere compatibili con json e con python

    raw_db_string = raw_db_string[0]
    raw_db_string = raw_db_string.replace("ObservedList(value=[", "[").replace("])", "]")  # da oggetto replit a elemento normale python
    raw_db_string = raw_db_string.replace("ObservedDict(value={", "{").replace("})", "}")

    # da oggetto pythona  oggetto json
    raw_db_string = raw_db_string.replace(" True, ", " true, ").replace(" False, ", " false, ").replace(" None, ", " null, ")
    raw_db_string = raw_db_string.replace(" True}", " true}").replace(" False}", " false}").replace(" None}", " null}")
    raw_db_string = raw_db_string.replace(" True]", " true]").replace(" False]", " false]").replace(" None]", " null]")

    # analisi della stringa che ci permette di capire quali sono le parti comprese in virgolette

    quoted_indexes = []
    in_quotes = False

    for i, char in enumerate(raw_db_string[1:]):
        i += 1

        if char in ["'", '"'] and raw_db_string[i-1] == "\\":  # se il carattere è una escaped quote non lo prendiamo in considerazione
            pass

        else:
            if char in ["'", '"'] and in_quotes == False:
                in_quotes = True
                quoted_indexes.append([i+1])
                beginning_quote = "single" if char == "'" else "double"

            elif in_quotes and ((char == "'" and beginning_quote == "single") or (char == '"' and beginning_quote == "double")):
                in_quotes = False
                quoted_indexes[-1].append(i)

    def double_quoted(string):
        """ 'abc \' "" '  ->  "abc" ' \"\" """
        new_string = """"""
        new_string += '"'
        new_string += string.replace("\\'", "'").replace('"', '\\"')
        new_string += '"'

        return new_string

    # riscrittura del db sottoforma di json per poi caricarlo come dizionario python
    db_string_jsonFormat = "{"
    last_idx = 0
    for range_ in quoted_indexes:
        db_string_jsonFormat += raw_db_string[last_idx+1:range_[0]-1]  # +1 e -1 tolgono le "" o '' che c'erano precedentemente
        db_string_jsonFormat += double_quoted(raw_db_string[range_[0]:range_[1]])
        last_idx = range_[1]
    db_string_jsonFormat += raw_db_string[last_idx+1:]  # aggingiamo il pezzo finale

    db_dict = json.loads(db_string_jsonFormat)
    return db_dict
