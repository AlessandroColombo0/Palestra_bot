# coding: utf8
import copy

from funzioni import ora_EU, ic

from creazione_immagini import days_to_mesiSett


a = 6
b = 3

if 2 < b < a:
    print(b)

7/0

class aa():
    def __init__(self, num):
        self.num = num
        self.a = self.motti(1)

    def motti(self, num):
        return num + 1

    def __str__(self):

        return f"aa object: {self.num}"

    def restart(self):
        globals()["a"] = aa(2)

a = aa(3)
ic(a)

a.restart()

ic(a)

def b():
    b = a
    ic(b)

    globals()["a"] = "c"

    return a




c = b()
ic(a)
ic(c)
7/0





def func():
    # global alala
    globals()["alala"] = 3
func()
ic(alala)



7/0

a = [1,2,3]
exec("a[1] = 4", )
ic(a)

db = {"1": {"2": 2, "3": [31, 32, 33]}}

def db_insert(indexers, val):
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
        print(err)
    else:
        indexing_str = "".join([f"['{i}']" if type(i) == str else f"[{i}]" for i in indexers])
        exec(f"db{indexing_str} = val", {"db": db, "val": val})


db_insert(["1", "3", 1], "aa")
ic(db)

a = {"1": 1}

a = b = 2 + 2
ic(a)
ic(b)

if 1 in a:
    print("SO")

1/0


m, s = days_to_mesiSett(10)
ic(m)
ic(s)

a = [1,2, 3, 4 ]
b = a[:3]
ic(b)

_ = 1
a = []
a.append(_)

ic(a)
a = ora_EU(isoformat=True)


a = ["a", "a"]
b = set(a)
c = set("a")

if c == b:
    print("S")

1/0




import shutil

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

import time

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
            dict_wrap_width = wrap_width - (len(key) + 5)
            new_value = element_wrap(dict_wrap_width, str(value), key=key)

            new_str += f'"{key}": {new_value[:-1]},\n '
        new_str = new_str[:-3] + "}"

    else:
        new_str = element_wrap(wrap_width, str(obj))

    return new_str
ic.configureOutput(prefix="> ", includeContext=True, argToStringFunction=to_string)


from plot_model import plot_model
import tensorflow as tf

model_name = "hpo_model"

model = tf.keras.models.load_model(f"msg labelling/{model_name}")
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, style=0, color=True, dpi=96)
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True, style=0, color=True, dpi=96)






stop = 1 / 0




import datetime as dt
import os
import json

def secs_to_hours(sec, show_seconds=False):
    """ es: secs_to_hours(11545) = 3h 12m 25s, secs_to_hours(600) = 10m 0s"""
    HH_MM_SS = str(dt.timedelta(seconds=sec))
    HMS_split = HH_MM_SS.split(":")
    print(HMS_split)

    # round dei minuti
    if not show_seconds:
        if float(HMS_split[2]) > 30:
            HMS_split[1] = str(int(HMS_split[1]) + 1)

    seconds = str(int(float(HMS_split[2]))) + "s" if show_seconds else ""
    mins = str(int(HMS_split[1])) + "m "
    ore = str(int(HMS_split[0])) + "h " if HMS_split[0] != "0" else ""
    # str(int()) serve per togliere gli 0, es: 08 -> 8

    return ore + mins + seconds

a = secs_to_hours(11545, True)
print(a)



import numpy as np
import tensorflow as tf
import random


idx_to_f = {0: "nomi", 1: "reps", 2: "peso", 3: "note"}
db = {"new_training_data": [["lat pull", idx_to_f[random.randint(0,3)]] for _ in range(69)],
      "models": {"hpo_model": {"name": "diocaneoid", "version": 1.}},
      "val_results": [0.0, 0.0]}
ic(db)


# TRAINING MODELLO
if len(db["new_training_data"]) > 50:
    start = time.perf_counter()

    model_name = "hpo_model"
    num_classes = 4
    tipi = ["nomi", "reps", "peso", "note"]
    # bot.send_message(a_uid, f"Allenamento di {db['models'][model_name]['name']} in corso...")

    # AGGIUNTA SAMPLES A TRAINING DATA
    TD_filePath = "msg labelling/training_data.json"
    with open(TD_filePath, "r") as file:
        trainig_data_json = json.load(file)

    num_samples = dict(nomi=[], reps=[], peso=[], note=[])  # [0] samples prima di aggiunta di new_training_data, [1] samples dopo
    [num_samples[tipo].append(len(samples)) for tipo, samples in trainig_data_json.items()]

    for text, key in db["new_training_data"]:
        trainig_data_json[key].append(text)

    with open(TD_filePath, "w") as file:
        json.dump(trainig_data_json, file, indent=4)

    # DATA PIPELINE
    one_hot_labels = dict(nomi=np.array([1., 0., 0., 0.]), reps=np.array([0., 1., 0., 0.]),
                          peso=np.array([0., 0., 1., 0.]), note=np.array([0., 0., 0., 1.]))


    def get_dataset(folder, training=True):
        """ oltre ad ottenere il ds calcola anche num_samples """
        with open(folder, "r") as file:
            trainig_data_json = json.load(file)

        read_txtFiles = {key: value for key, value in trainig_data_json.items()}

        text_list = []
        oneHot_list = []
        for tipo, samples in read_txtFiles.items():
            text_list.extend(samples)
            oneHot_list.extend([one_hot_labels[tipo] for _ in samples])
            if training:

                num_samples[tipo].append(len(samples))

        X_ds, Y_ds = tf.data.Dataset.from_tensor_slices(text_list), tf.data.Dataset.from_tensor_slices(oneHot_list)
        dataset = tf.data.Dataset.zip((X_ds, Y_ds)).shuffle(len(text_list)).batch(32)

        return dataset, oneHot_list

    training_ds, train_oneHots = get_dataset("msg labelling/training_data.json")
    testing_ds, _ = get_dataset("msg labelling/training_data.json")

    # TRAINING
    # creaizone class weights per training bilanciato
    from sklearn.utils.class_weight import compute_class_weight
    train_labels_idxs = [np.argmax(np.array(lista)) for lista in train_oneHots]  # tutti gli one hots sottoforma di indexes
    class_weights = compute_class_weight(class_weight="balanced", classes=list(range(num_classes)), y=train_labels_idxs)
    class_weights = {i: weight for i, weight in enumerate(class_weights)}

    n_epochs = 1
    lr = 0.0005
    label_smoothing = 0.2

    model = tf.keras.models.load_model(f"msg labelling/{model_name}")
    model.compile(tf.optimizers.Adam(learning_rate=lr), tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), "accuracy")

    fit = model.fit(training_ds, epochs=n_epochs, class_weight=class_weights)
    # model.save(f"msg labelling/{model_name}", save_format="tf")
    eval = model.evaluate(testing_ds)

    # SUMMARY
    ns = num_samples
    ns_d = {tipo: ns[tipo][-1] - ns[tipo][0] for tipo in tipi}
    total = sum([ns[tipo][-1] for tipo in tipi])
    ns_p = {tipo: ns[tipo][-1]/total for tipo in tipi}
    E, R, P, N = "nomi", "reps", "peso", "note"
    val_diff = [eval[0] - db["val_results"][0], eval[-1] - db["val_results"][-1]]

    end = time.perf_counter()

    summary_msg = f"""
Tempo impiegato: {secs_to_hours(end-start, True)}
    
*Dataset*
 new samples: {len(db["new_training_data"])}
 `E: {ns[N][-1]}, {ns_p[N]:.0%} ({ns_d[N]:+.0f})  R: {ns[R][-1]}, {ns_p[R]:.0%} ({ns_d[R]:+.0f})` 
 `P: {ns[P][-1]}, {ns_p[P]:.0%} ({ns_d[P]:+.0f})  N: {ns[N][-1]}, {ns_p[R]:.0%} ({ns_d[P]:+.0f})`

*Training*
 class weights: (`E: {class_weights[0]:.3f}, R: {class_weights[1]:.3f}, P: {class_weights[2]:.3f}, N: {class_weights[3]:.3f}`)
 epochs: {n_epochs}, lr: {lr}, label smoothing: {label_smoothing}

*Results*
 `train: A: {fit.history["accuracy"][-1]:.2%}            L: {fit.history["loss"][-1]:.3f}` 
 `test:  A: {eval[-1]:.2%} ({val_diff[-1]:+.2%})  L: {eval[0]:.3f} ({val_diff[0]:.3f})`
 ⚠️ prendere nota del fatto che il dataset di testing è abbastanza lontano da quello di training e non ha reps e peso.
"""
    summary_msg = md_v2_replace(summary_msg)
    print(summary_msg)
    stop = 1 / 0

    db["val_results"] = [eval[0], eval[-1]]
    db["models"][model_name]["version"] = db["models"][model_name]["version"] + 1
    db["new_training_data"] = []

    zipped_model_name = f"temporanei/{db['models'][model_name]['name']} V{db['models'][model_name]['version']}"
    shutil.make_archive(zipped_model_name, 'zip', model_name)
    for file in [zipped_model_name, "msg_labelling/training_data.json", "msg_labelling/testing_data.json"]:
        loaded_f = open(file, "rb")
        # bot.send_message()




#
# read_txtFiles = dict(nomi=open("msg labelling/training data/nomi"), reps=open("msg labelling/training data/reps"), peso=open("msg labelling/training data/peso"), note=open("msg labelling/training data/note"))
# list_txtFiles = {}
# for key, val in read_txtFiles.items():
#     list_txtFiles[key] = [i.strip() for i in val.readlines()]
# ic(list_txtFiles)
#
# # ic(e)
# # ic(b)
#
#
# a = [1,2,3]
# a.insert(3, 4)
# ic(a)
#
#
#
# # a = ["1"]
# # b = ["2"]
# # c = a + b
# # print(c)
#
# stop = 1 / 0
#
#
# # todo creare una funzione per inserire elementi nel database che dia errore se si cerca di inserire nel database un set, una stringa ocn "])", e altri elementi che mettono in
# #   confusione il replace
#
# string = """{'allenamenti': [ObservedList(value=['2021-05-01T00:00:00', 'pettobi']), ObservedList(value=['2021-06-02T00:00:00', 'pettobi']), ObservedList(value=['2022-06-28T22:03:00.349815', 'pettobi']), ObservedList(value=['2022-06-30T22:02:27.841406', 'spalletri']), ObservedList(value=['2022-07-02T22:06:41.745507', 'pettobi', 'addominali']), ObservedList(value=['2022-07-04T22:04:21.529954', 'schiena']), ObservedList(value=['2022-07-06T16:10:49.662584', 'pettobi', 'addominali']), ObservedList(value=['2022-07-28T22:05:16.593999', 'spalletri', 'pettobi']), ObservedList(value=['2022-08-02T22:00:10.786688', 'addominali']), ObservedList(value=['2022-08-04T22:07:50.956170', 'schiena']), ObservedList(value=['2022-08-06T22:01:29.117742', 'spalletri', 'addominali']), ObservedList(value=['2022-08-08T22:07:18.017165', 'pettobi']), ObservedList(value=['2022-08-09T22:09:14.057384', 'schiena', 'addominali']), ObservedList(value=['2022-08-10T22:06:10.398188', 'spalletri']), ObservedList(value=['2022-08-12T22:02:55.719616', 'addominali', 'pettobi']), ObservedList(value=['2022-08-17T22:05:24.007626', 'schiena']), ObservedList(value=['2022-08-18T22:04:49.278957', 'spalletri']), ObservedList(value=['2022-08-19T22:01:25.444301', 'addominali']), ObservedList(value=['2022-08-22T22:06:57.185344', 'pettobi', 'addominali']), ObservedList(value=['2022-08-25T22:09:46.849795', 'schiena', 'addominali']), ObservedList(value=['2022-08-26T22:04:57.266191', 'spalletri']), ObservedList(value=['2022-08-29T22:05:55.810576', 'pettobi', 'addominali']), ObservedList(value=['2022-08-30T22:07:04.340869', 'schiena']), ObservedList(value=['2022-08-31T22:06:21.858619', 'spalletri']), ObservedList(value=['2022-09-02T22:07:08.919972', 'addominali']), ObservedList(value=['2022-10-01T22:04:55.202453', 'spalletri', 'addominali']), ObservedList(value=['2022-10-04T22:00:05.545632', 'pettobi', 'addominali']), ObservedList(value=['2022-10-07T22:02:02.631164', 'schiena', 'addominali']), ObservedList(value=['2022-10-08T22:09:43.564880', 'spalletri']), ObservedList(value=['2022-10-10T22:09:54.371073', 'addominali', 'pettobi']), ObservedList(value=['2022-10-12T22:03:04.374646', 'schiena']), ObservedList(value=['2022-10-14T22:07:21.647547', 'spalletri', 'addominali']), ObservedList(value=['2022-10-17T22:09:19.897295', 'pettobi', 'addominali']), ObservedList(value=['2022-10-20T22:04:01.201494', 'schiena', 'addominali']), ObservedList(value=['2022-10-21T22:00:50.981645', 'spalletri']), ObservedList(value=['2022-10-24T22:03:29.088096', 'pettobi', 'addominali']), ObservedList(value=['2022-11-04T23:06:03.019767', 'spalletri']), ObservedList(value=['2022-12-07T19:40:45.994393', 'addominali', 'pettobi']), ObservedList(value=['2022-12-20T23:05:33.345562', 'schiena']), ObservedList(value=['2022-12-27T23:01:32.130200', 'pettobi', 'addominali']), ObservedList(value=['2022-12-29T23:06:19.523063', 'schiena']), ObservedList(value=['2022-12-31T23:04:35.835921', 'spalletri', 'addominali']), ObservedList(value=['2023-01-02T23:06:53.067975', 'pettobi']), ObservedList(value=['2023-01-03T23:04:58.327716', 'schiena', 'addominali']), ObservedList(value=['2023-02-20T22:22:25.525690', 'spalletri']), ObservedList(value=['2023-02-22T22:04:40.901145', 'pettobi', 'addominali']), ObservedList(value=['2023-02-23T23:14:16.755518', 'schiena']), ObservedList(value=['2023-02-28T22:23:31.470255', 'addominali']), ObservedList(value=['2023-03-02T22:44:42.104510', 'schiena'])], 'aumenti_peso': [ObservedList(value=['2021-05-01T00:00:00', '34k', '36k', 'Crunch']), ObservedList(value=['2021-06-02T00:00:00', '34k', '36k', 'Crunch']), ObservedList(value=['2022-07-04T17:00:11.704611', '25㎏', 'mezzi stacchi', '30㎏']), ObservedList(value=['2022-07-04T17:31:13.740570', '25㎏ \\+ 2 \\(\\- 2\\) …', 'lat pull maniglie', '32㎏ + 2 (- 2) …']), ObservedList(value=['2022-07-08T16:44:13.923236', '10㎏', 'lento avanti', '10㎏ 2,5㎏']), ObservedList(value=['2022-07-11T17:35:58.054621', '32㎏; 18㎏', 'Curl BD', '36㎏; 14㎏']), ObservedList(value=['2022-08-08T11:41:33.587848', '6㎏', 'Curl spider 30°', '7㎏']), ObservedList(value=['2022-08-09T19:47:44.876065', '32㎏ \\+ 2 \\(\\- 2\\) …', 'lat pull maniglie', '25㎏ + 2 (- 2) …']), ObservedList(value=['2022-08-18T12:35:05.151178', '9㎏ \\+ 2 \\(\\- 2\\) …', 'shoulder press', '9㎏ + 2 (- 2) …']), ObservedList(value=['2022-09-02T18:08:27.981268', '18㎏', 'Spinte p 30°', '20㎏']), ObservedList(value=['2022-09-03T11:40:43.422265', '23㎏ \\- 18㎏ \\- 14㎏', 'rem cavo basso corda', '27㎏ - 23㎏ - 18㎏']), ObservedList(value=['2022-10-01T12:05:07.305885', '10㎏ 2,5㎏', '15㎏', 'Lento avanti']), ObservedList(value=['2022-10-04T18:31:28.617643', '25㎏ \\+ 2', '32㎏ + 1', 'Chest press']), ObservedList(value=['2022-10-07T17:26:17.375654', '27㎏ \\- 23㎏ \\- 18㎏', '32㎏ - 27㎏ - 23㎏', 'Rem cavo basso corda']), ObservedList(value=['2022-10-10T17:09:36.769367', '16㎏', '18㎏', 'Spinte pp']), ObservedList(value=['2022-10-14T17:10:16.045986', '18㎏; 9㎏', '23㎏: 9㎏', 'Alzate front bd']), ObservedList(value=['2022-10-14T17:29:59.446353', '6㎏', '7㎏', 'Alzate lat manubri']), ObservedList(value=['2022-10-17T17:38:49.432628', '18㎏; 14㎏ \\- …', '18㎏; 18㎏ - ...', 'Croci ai cavi']), ObservedList(value=['2022-10-17T17:48:00.538771', '20㎏', '22㎏', 'Spinte p 30°']), ObservedList(value=['2022-10-20T17:48:10.045238', '14㎏ \\+ 2', '18㎏ + 2', 'Ss diverging seat row: orizz \\| vertic']), ObservedList(value=['2022-10-20T17:48:23.734082', '25㎏ \\+ 2 \\(\\- 2\\) …', '32㎏ + 2 (- 2) …', 'Lat pull maniglie']), ObservedList(value=['2022-10-21T17:37:09.588936', '9㎏ \\+ 2 \\(\\- 2\\) …', '9㎏ + 2 (- 1) …', 'Shoulder press']), ObservedList(value=['2022-11-04T18:21:55.749141', '7㎏', '8㎏', 'Alzate laterali']), ObservedList(value=['2022-11-07T18:32:15.220871', '16㎏, 12㎏, 8㎏', '16㎏, 10㎏, 6㎏', 'Distensioni man\\. p:1']), ObservedList(value=['2022-12-20T19:34:50.953011', '18㎏ \\+ 1, 23㎏', '18㎏ + 1, 27㎏', 'Lat pull / pull down / lat pull']), ObservedList(value=['2022-12-23T18:09:55.695725', '16㎏ / 6㎏', '18㎏ / 6㎏', 'Lento man\\. / arnold press']), ObservedList(value=['2022-12-23T19:10:23.128224', '5㎏ \\(\\+ 1,5㎏\\)', '7,5㎏ (+ 1,25㎏)', 'French press ez']), ObservedList(value=['2022-12-29T18:11:13.082495', '18㎏ \\+ 1, 27㎏', '18㎏ + 2, 27㎏', 'Lat pull / pull down / lat pull']), ObservedList(value=['2023-02-20T19:35:03.746611', '14㎏, 9㎏', '18㎏, 9㎏', 'Alzate frontali \\(corda\\)']), ObservedList(value=['2023-02-20T20:02:54.721372', '41㎏; 36㎏, 45㎏', '32㎏, 27㎏, 36㎏', 'Push down bd']), ObservedList(value=['2023-02-20T20:03:35.609091', '41㎏; 36㎏, 45㎏', '32㎏; 27㎏, 36㎏', 'Push down bd']), ObservedList(value=['2023-02-22T17:19:31.603513', '26㎏', '28㎏', 'Distensioni man\\. 45°']), ObservedList(value=['2023-02-22T17:42:20.390988', '12㎏', '14㎏', 'Croci pp']), ObservedList(value=['2023-02-22T17:52:49.530590', '14㎏', '16㎏', 'Croci panca declinata 30°']), ObservedList(value=['2023-02-24T17:39:57.428096', '8㎏', '10㎏', 'Alzate laterali']), ObservedList(value=['2023-02-24T18:08:26.556634', '4㎏', '5㎏', 'Alzate lat\\. \\+ front\\. \\(comb\\)']), ObservedList(value=['2023-03-01T18:42:25.837480', '36㎏ \\(\\- 1\\)', '36㎏ + 2', 'Low row \\(pr\\. p\\.\\)']), ObservedList(value=['2023-03-01T18:42:55.818805', '36㎏ \\(\\- 1\\)', '36㎏ + 2 (- 1)', 'Low row \\(pr\\. p\\.\\)']), ObservedList(value=['2023-03-01T19:07:49.657814', '39㎏', '39㎏ + 2', 'Lat pull']), ObservedList(value=['2023-03-01T19:08:24.305976', '15㎏', '20㎏', 'Dante row']), ObservedList(value=['2023-03-06T17:43:25.358682', '8㎏', '10㎏', 'Alzate laterali']), ObservedList(value=['2023-03-06T18:02:26.532864', '14㎏, 9㎏', '18㎏, 14㎏', 'Alzate frontali \\(corda\\)']), ObservedList(value=['2023-03-06T18:16:36.805253', '36㎏ \\+ 1 \\(\\-1\\)', '41㎏ (-1)', 'Shoulder press'])], 'calendario': [2023, 2], 'Cycle_4settLever': True, 'Cycle_dailyLever': True, 'Cycle_monthLever': True, 'data_2volte': {'pettobi': None, 'spalletri': None, 'addominali': None, 'schiena': None, 'altro': None}, 'data_cambioscheda': ['2022-11-03T21:51:40.118463', '2023-02-08T00:10:31.600859'], 'data_prossima_stat': '2023-03-06T00:00:00', 'data_prossima_stat_m': [2023, 2, 28], 'dizAll_2volte': {'pettobi': None, 'spalletri': None, 'addominali': None, 'schiena': None, 'altro': None}, 'fine_allenamenti': {'addominali': ObservedList(value=['`6 Mar`\nTempo impiegato: *18m 37s*\nDalle 18:42 alle 19:01', '`28 Feb`\nTempo impiegato: *00m 41s*\nDalle 19:21 alle 19:22', '`22 Feb`\nTempo impiegato: *22m 15s*\nDalle 18:18 alle 18:40', '`3 Gen`\nTempo impiegato: *25m 31s*\nDalle 17:48 alle 18:14']), 'pettobi': ObservedList(value=['`22 Feb`\nTempo impiegato: *1h 07m 40s*\nDalle 17:06 alle 18:14', '`2 Gen`\nTempo impiegato: *1h 09m 59s*\nDalle 17:28 alle 18:38', '`27 Dic`\nTempo impiegato: *1h 14m 02s*\nDalle 17:48 alle 19:02', '`7 Nov`\nTempo impiegato: *N/A*\nFine: 19:23']), 'schiena': ObservedList(value=['`1 Mar`\nTempo impiegato: *55m 29s*\nDalle 18:19 alle 19:15', '`23 Feb`\nTempo impiegato: *55m 04s*\nDalle 17:28 alle 18:23', '`3 Gen`\nTempo impiegato: *52m 55s*\nDalle 16:50 alle 17:43', '`29 Dic`\nTempo impiegato: *47m 20s*\nDalle 17:10 alle 17:58']), 'spalletri': ObservedList(value=['`6 Mar`\nTempo impiegato: *58m 26s*\nDalle 17:40 alle 18:38', '`20 Feb`\nTempo impiegato: *49m 11s*\nDalle 19:16 alle 20:05', '`31 Dic`\nTempo impiegato: *54m 18s*\nDalle 12:19 alle 13:13', '`4 Nov`\nTempo impiegato: *N/A*\nFine: 18:06']), 'altro': ObservedList(value=['addominali, 28/6, 1ora', 'addominali, 28/6, 1ora', 'addominali, 28/6, 1ora', 'addominali, 28/6, 1ora'])}, 'last_msg_id': 36520, 'latest_messageId': 35836, 'latest_msgId': 36001, 'lever_cambioscheda': False, 'lever_thread_4sett': True, 'lever_thread_mens': True, 'lever_ultimi_allenamenti': True, 'peso_bilancia': [ObservedList(value=['2023-01-02T19:15:59.882048', 70.85]), ObservedList(value=['2023-01-10T19:29:20.740950', 70.9]), ObservedList(value=['2023-01-17T20:06:44.899653', 70.65]), ObservedList(value=['2023-01-19T19:57:53.002134', 70.6]), ObservedList(value=['2023-01-24T19:52:14.165599', 71.2]), ObservedList(value=['2023-01-26T20:06:03.379746', 71.25]), ObservedList(value=['2023-01-31T19:58:35.247931', 70.4]), ObservedList(value=['2023-02-23T19:49:04.873234', 71.3]), ObservedList(value=['2023-02-24T19:56:24.236216', 72.1]), ObservedList(value=['2023-02-28T19:43:58.971291', 72.0])], 'preserved_msgIds': [36125, 36130, 36406, 36450, 36486], 'prev_tempo': {'addominali': ObservedList(value=[ObservedList(value=[156.05405564900138, 111.02994525800023, 99.94005111100341, 97.61069813799986, 97.9964455670015, 91.4806724730006, 105.48080302200106, 107.126901636002, 125.81684930399933, 80.01495901100134, 80.01495901100134]), ObservedList(value=[1.3476661240056274, 9.221072176995222, 112.67035331399529, 107.0965926219942, 140.38410791000206, 91.74315100700187, 76.9428340010054, 86.79510790500353, 8.745087525996496, 86.42529395500605, 86.42529395500605]), ObservedList(value=[100, 100, 100, 100, 85, 85, 85, 85, 70, 70, 70])]), 'pettobi': ObservedList(value=[ObservedList(value=[456.10524176499894, 127.79336847100058, 122.35872185799963, 132.28754203299468, 147.63920145599695, 161.25068290800118, 129.06809857599728, 112.81179034700472, 131.97456286700617, 119.28798122199805, 103.0892560909997, 235.1453804970006, 139.95653933100402, 108.76585689099738, 119.87495730200317, 152.14284440499614, 122.09336197198718, 150.29440029599937, 130.24911408800108, 124.33360110099602, 126.63049956699251, 169.92522055598965, 147.38983448999352, 137.62194637001085, 137.30152714699216, 137.9003726439987, 103.79556352099462, 103.79556352099462]), ObservedList(value=[120, 120, 120, 120, 120, 85, 85, 85, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100]), ObservedList(value=[120, 120, 120, 120, 120, 85, 85, 85, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100])]), 'schiena': ObservedList(value=[ObservedList(value=[42.015306240005884, 85.75019623599655, 91.98791161400004, 82.8455477450043, 285.3548354779996, 81.3753198580016, 131.81698198999948, 140.6067977929997, 138.3048929200013, 156.83368331399834, 155.82047453699488, 226.18813510699692, 144.87260614000115, 144.2630102980038, 166.81588091800222, 289.37823802800267, 186.508419951002, 198.28363388300204, 262.6716593559977, 138.36426698500145, 138.36426698500145]), ObservedList(value=[66.4217054159999, 95.77462775499953, 90.79615308199936, 103.58599200900062, 301.89677512600065, 139.88165118100005, 148.89739370000098, 129.66158106199873, 136.42410586799997, 263.640988915, 180.0425736039997, 194.78876624400073, 118.48784315400007, 152.84040112399998, 144.60361823899984, 200.59278008600086, 186.69544120599858, 174.27646199799892, 260.0723366080001, 120.81669442099883, 120.81669442099883]), ObservedList(value=[100, 100, 100, 100, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120])]), 'spalletri': ObservedList(value=[ObservedList(value=[180.1448306099992, 83.7315916439984, 85.42463757400037, 86.7639201940001, 90.68764061000184, 291.47758539100323, 11.297743769002409, 148.08050215500043, 133.9215232020033, 118.19201238900132, 161.45808495700112, 108.85746658500284, 109.39056832200004, 133.10755225499815, 139.76460772399878, 110.35530535900034, 120.04980469300062, 249.4628320970005, 198.74998035599856, 118.34003289799875, 135.9024515400015, 134.88953536899862, 287.2228129690011, 99.70586600000024, 94.9509411830004, 94.9509411830004]), ObservedList(value=[5.504287674004445, 96.42483057100617, 96.22209244199621, 96.81179361999966, 86.52368478999415, 106.5325201079977, 158.69735562300775, 131.00018851298955, 121.5029678150022, 117.26818278399878, 127.83320070699847, 111.93345461699937, 114.4314488000091, 117.88109284300299, 113.12073416999192, 115.5502365949942, 117.78472933599551, 226.56726013100706, 165.65748278099636, 132.3947183470009, 134.64073086999997, 132.45129953000287, 97.23156634399493, 97.3312075290014, 94.67455334600527, 94.67455334600527]), ObservedList(value=[100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100, 100])]), 'altro': ObservedList(value=[ObservedList(value=[]), ObservedList(value=[]), ObservedList(value=[])])}, 'test_dict': {'a': ObservedDict(value={'b': ObservedList(value=[ObservedList(value=[1, 2, 3]), ObservedList(value=[1, 2, 3])])})}, 'tracebacks': {}, 'ultima_stat': {'y_aumenti_peso': 0, 'y_allenamenti': 0, 'y_addominali': 0}, 'ultimi_allenamenti': [ObservedList(value=["*Schiena*, `{'gio'} 23 Feb `", '*Addominali*, `mar 28 Feb `', '*Schiena*, `gio 02 Mar `']), ObservedList(value=['2023-02-23T23:14:16.755518', '2023-02-28T22:23:31.470255', '2023-03-02T22:44:42.104510'])], 'ultimo_allenamento': ['spalletri', 'addominali'], 'workout_count': {'pettobi': 0, 'spalletri': 0, 'addominali': 0, 'schiena': 0, 'altro': 0}}"""
#
# db_keys = ['allenamenti', 'aumenti_peso', 'calendario', 'Cycle_4settLever', 'Cycle_dailyLever', 'Cycle_monthLever', 'data_2volte', 'data_cambioscheda', 'data_prossima_stat', 'data_prossima_stat_m', 'dizAll_2volte', 'fine_allenamenti', 'last_msg_id', 'latest_messageId', 'latest_msgId', 'lever_cambioscheda', 'lever_thread_4sett', 'lever_thread_mens', 'lever_ultimi_allenamenti', 'peso_bilancia', 'preserved_msgIds', 'prev_tempo', 'tracebacks', 'ultima_stat', 'ultimi_allenamenti', 'ultimo_allenamento', 'workout_count']
#
# with open('raw_db.txt') as f:
#     text_file = f.readlines()
#
# print(text_file)
#
# t_string = ", 'test_dict': {'a': ObservedDict(value={'b': ObservedList(value=[ObservedList(value=['sldkn)]vd}]l', 2, 3]), ObservedList(value=[1, 2, 3])])})}, "
# t_string_replace = t_string.replace("ObservedList(value=[", "[").replace("ObservedDict(value={", "{")
# print(t_string_replace)
#
#
# string = string.replace("ObservedList(value=[", "[").replace("])", "]")
# string = string.replace("ObservedDict(value={", "{").replace("})", "}")
# print(string)
#
#
#
#
# stop = 1 / 0
#
# """{'allenamenti': [['2021-05-01T00:00:00', 'pettobi'], ['2021-06-02T00:00:00', 'pettobi'], ['2022-06-28T22:03:00.349815', 'pettobi'], ['2022-06-30T22:02:27.841406', 'spalletri'], ['2022-07-02T22:06:41.745507', 'pettobi', 'addominali'], ['2022-07-04T22:04:21.529954', 'schiena'], ['2022-07-06T16:10:49.662584', 'pettobi', 'addominali'], ['2022-07-28T22:05:16.593999', 'spalletri', 'pettobi'], ['2022-08-02T22:00:10.786688', 'addominali'], ['2022-08-04T22:07:50.956170', 'schiena'], ['2022-08-06T22:01:29.117742', 'spalletri', 'addominali'], ['2022-08-08T22:07:18.017165', 'pettobi'], ['2022-08-09T22:09:14.057384', 'schiena', 'addominali'], ['2022-08-10T22:06:10.398188', 'spalletri'], ['2022-08-12T22:02:55.719616', 'addominali', 'pettobi'], ['2022-08-17T22:05:24.007626', 'schiena'], ['2022-08-18T22:04:49.278957', 'spalletri'], ['2022-08-19T22:01:25.444301', 'addominali'], ['2022-08-22T22:06:57.185344', 'pettobi', 'addominali'], ['2022-08-25T22:09:46.849795', 'schiena', 'addominali'], ['2022-08-26T22:04:57.266191', 'spalletri'], ['2022-08-29T22:05:55.810576', 'pettobi', 'addominali'], ['2022-08-30T22:07:04.340869', 'schiena'], ['2022-08-31T22:06:21.858619', 'spalletri'], ['2022-09-02T22:07:08.919972', 'addominali'], ['2022-10-01T22:04:55.202453', 'spalletri', 'addominali'], ['2022-10-04T22:00:05.545632', 'pettobi', 'addominali'], ['2022-10-07T22:02:02.631164', 'schiena', 'addominali'], ['2022-10-08T22:09:43.564880', 'spalletri'], ['2022-10-10T22:09:54.371073', 'addominali', 'pettobi'], ['2022-10-12T22:03:04.374646', 'schiena'], ['2022-10-14T22:07:21.647547', 'spalletri', 'addominali'], ['2022-10-17T22:09:19.897295', 'pettobi', 'addominali'], ['2022-10-20T22:04:01.201494', 'schiena', 'addominali'], ['2022-10-21T22:00:50.981645', 'spalletri'], ['2022-10-24T22:03:29.088096', 'pettobi', 'addominali'], ['2022-11-04T23:06:03.019767', 'spalletri'], ['2022-12-07T19:40:45.994393', 'addominali', 'pettobi'], ['2022-12-20T23:05:33.345562', 'schiena'], ['2022-12-27T23:01:32.130200', 'pettobi', 'addominali'], ['2022-12-29T23:06:19.523063', 'schiena'], ['2022-12-31T23:04:35.835921', 'spalletri', 'addominali'], ['2023-01-02T23:06:53.067975', 'pettobi'], ['2023-01-03T23:04:58.327716', 'schiena', 'addominali'], ['2023-02-20T22:22:25.525690', 'spalletri'], ['2023-02-22T22:04:40.901145', 'pettobi', 'addominali'], ['2023-02-23T23:14:16.755518', 'schiena'], ['2023-02-28T22:23:31.470255', 'addominali'], ['2023-03-02T22:44:42.104510', 'schiena']], 'aumenti_peso': [['2021-05-01T00:00:00', '34k', '36k', 'Crunch'], ['2021-06-02T00:00:00', '34k', '36k', 'Crunch'], ['2022-07-04T17:00:11.704611', '25㎏', 'mezzi stacchi', '30㎏'], ['2022-07-04T17:31:13.740570', '25㎏ \+ 2 \(\- 2\) …', 'lat pull maniglie', '32㎏ + 2 (- 2) …'], ['2022-07-08T16:44:13.923236', '10㎏', 'lento avanti', '10㎏ 2,5㎏'], ['2022-07-11T17:35:58.054621', '32㎏; 18㎏', 'Curl BD', '36㎏; 14㎏'], ['2022-08-08T11:41:33.587848', '6㎏', 'Curl spider 30°', '7㎏'], ['2022-08-09T19:47:44.876065', '32㎏ \+ 2 \(\- 2\) …', 'lat pull maniglie', '25㎏ + 2 (- 2) …'], ['2022-08-18T12:35:05.151178', '9㎏ \+ 2 \(\- 2\) …', 'shoulder press', '9㎏ + 2 (- 2) …'], ['2022-09-02T18:08:27.981268', '18㎏', 'Spinte p 30°', '20㎏'], ['2022-09-03T11:40:43.422265', '23㎏ \- 18㎏ \- 14㎏', 'rem cavo basso corda', '27㎏ - 23㎏ - 18㎏'], ['2022-10-01T12:05:07.305885', '10㎏ 2,5㎏', '15㎏', 'Lento avanti'], ['2022-10-04T18:31:28.617643', '25㎏ \+ 2', '32㎏ + 1', 'Chest press'], ['2022-10-07T17:26:17.375654', '27㎏ \- 23㎏ \- 18㎏', '32㎏ - 27㎏ - 23㎏', 'Rem cavo basso corda'], ['2022-10-10T17:09:36.769367', '16㎏', '18㎏', 'Spinte pp'], ['2022-10-14T17:10:16.045986', '18㎏; 9㎏', '23㎏: 9㎏', 'Alzate front bd'], ['2022-10-14T17:29:59.446353', '6㎏', '7㎏', 'Alzate lat manubri'], ['2022-10-17T17:38:49.432628', '18㎏; 14㎏ \- …', '18㎏; 18㎏ - ...', 'Croci ai cavi'], ['2022-10-17T17:48:00.538771', '20㎏', '22㎏', 'Spinte p 30°'], ['2022-10-20T17:48:10.045238', '14㎏ \+ 2', '18㎏ + 2', 'Ss diverging seat row: orizz \| vertic'], ['2022-10-20T17:48:23.734082', '25㎏ \+ 2 \(\- 2\) …', '32㎏ + 2 (- 2) …', 'Lat pull maniglie'], ['2022-10-21T17:37:09.588936', '9㎏ \+ 2 \(\- 2\) …', '9㎏ + 2 (- 1) …', 'Shoulder press'], ['2022-11-04T18:21:55.749141', '7㎏', '8㎏', 'Alzate laterali'], ['2022-11-07T18:32:15.220871', '16㎏, 12㎏, 8㎏', '16㎏, 10㎏, 6㎏', 'Distensioni man\. p:1'], ['2022-12-20T19:34:50.953011', '18㎏ \+ 1, 23㎏', '18㎏ + 1, 27㎏', 'Lat pull / pull down / lat pull'], ['2022-12-23T18:09:55.695725', '16㎏ / 6㎏', '18㎏ / 6㎏', 'Lento man\. / arnold press'], ['2022-12-23T19:10:23.128224', '5㎏ \(\+ 1,5㎏\)', '7,5㎏ (+ 1,25㎏)', 'French press ez'], ['2022-12-29T18:11:13.082495', '18㎏ \+ 1, 27㎏', '18㎏ + 2, 27㎏', 'Lat pull / pull down / lat pull'], ['2023-02-20T19:35:03.746611', '14㎏, 9㎏', '18㎏, 9㎏', 'Alzate frontali \(corda\)'], ['2023-02-20T20:02:54.721372', '41㎏; 36㎏, 45㎏', '32㎏, 27㎏, 36㎏', 'Push down bd'], ['2023-02-20T20:03:35.609091', '41㎏; 36㎏, 45㎏', '32㎏; 27㎏, 36㎏', 'Push down bd'], ['2023-02-22T17:19:31.603513', '26㎏', '28㎏', 'Distensioni man\. 45°'], ['2023-02-22T17:42:20.390988', '12㎏', '14㎏', 'Croci pp'], ['2023-02-22T17:52:49.530590', '14㎏', '16㎏', 'Croci panca declinata 30°'], ['2023-02-24T17:39:57.428096', '8㎏', '10㎏', 'Alzate laterali'], ['2023-02-24T18:08:26.556634', '4㎏', '5㎏', 'Alzate lat\. \+ front\. \(comb\)'], ['2023-03-01T18:42:25.837480', '36㎏ \(\- 1\)', '36㎏ + 2', 'Low row \(pr\. p\.\)'], ['2023-03-01T18:42:55.818805', '36㎏ \(\- 1\)', '36㎏ + 2 (- 1)', 'Low row \(pr\. p\.\)'], ['2023-03-01T19:07:49.657814', '39㎏', '39㎏ + 2', 'Lat pull'], ['2023-03-01T19:08:24.305976', '15㎏', '20㎏', 'Dante row'], ['2023-03-06T17:43:25.358682', '8㎏', '10㎏', 'Alzate laterali'], ['2023-03-06T18:02:26.532864', '14㎏, 9㎏', '18㎏, 14㎏', 'Alzate frontali \(corda\)'], ['2023-03-06T18:16:36.805253', '36㎏ \+ 1 \(\-1\)', '41㎏ (-1)', 'Shoulder press']], 'calendario': [2023, 2], 'Cycle_4settLever': True, 'Cycle_dailyLever': True, 'Cycle_monthLever': True, 'data_2volte': {'pettobi': None, 'spalletri': None, 'addominali': None, 'schiena': None, 'altro': None}, 'data_cambioscheda': ['2022-11-03T21:51:40.118463', '2023-02-08T00:10:31.600859'], 'data_prossima_stat': '2023-03-06T00:00:00', 'data_prossima_stat_m': [2023, 2, 28], 'dizAll_2volte': {'pettobi': None, 'spalletri': None, 'addominali': None, 'schiena': None, 'altro': None}, 'fine_allenamenti': {'addominali': ['`6 Mar`
# Tempo impiegato: *18m 37s*
# Dalle 18:42 alle 19:01', '`28 Feb`
# Tempo impiegato: *00m 41s*
# Dalle 19:21 alle 19:22', '`22 Feb`
# Tempo impiegato: *22m 15s*
# Dalle 18:18 alle 18:40', '`3 Gen`
# Tempo impiegato: *25m 31s*
# Dalle 17:48 alle 18:14'], 'pettobi': ['`22 Feb`
# Tempo impiegato: *1h 07m 40s*
# Dalle 17:06 alle 18:14', '`2 Gen`
# Tempo impiegato: *1h 09m 59s*
# Dalle 17:28 alle 18:38', '`27 Dic`
# Tempo impiegato: *1h 14m 02s*
# Dalle 17:48 alle 19:02', '`7 Nov`
# Tempo impiegato: *N/A*
# Fine: 19:23'], 'schiena': ['`1 Mar`
# Tempo impiegato: *55m 29s*
# Dalle 18:19 alle 19:15', '`23 Feb`
# Tempo impiegato: *55m 04s*
# Dalle 17:28 alle 18:23', '`3 Gen`
# Tempo impiegato: *52m 55s*
# Dalle 16:50 alle 17:43', '`29 Dic`
# Tempo impiegato: *47m 20s*
# Dalle 17:10 alle 17:58'], 'spalletri': ['`6 Mar`
# Tempo impiegato: *58m 26s*
# Dalle 17:40 alle 18:38', '`20 Feb`
# Tempo impiegato: *49m 11s*
# Dalle 19:16 alle 20:05', '`31 Dic`
# Tempo impiegato: *54m 18s*
# Dalle 12:19 alle 13:13', '`4 Nov`
# Tempo impiegato: *N/A*
# Fine: 18:06'], 'altro': ['addominali, 28/6, 1ora', 'addominali, 28/6, 1ora', 'addominali, 28/6, 1ora', 'addominali, 28/6, 1ora']}, 'last_msg_id': 36520, 'latest_messageId': 35836, 'latest_msgId': 36001, 'lever_cambioscheda': False, 'lever_thread_4sett': True, 'lever_thread_mens': True, 'lever_ultimi_allenamenti': True, 'peso_bilancia': [['2023-01-02T19:15:59.882048', 70.85], ['2023-01-10T19:29:20.740950', 70.9], ['2023-01-17T20:06:44.899653', 70.65], ['2023-01-19T19:57:53.002134', 70.6], ['2023-01-24T19:52:14.165599', 71.2], ['2023-01-26T20:06:03.379746', 71.25], ['2023-01-31T19:58:35.247931', 70.4], ['2023-02-23T19:49:04.873234', 71.3], ['2023-02-24T19:56:24.236216', 72.1], ['2023-02-28T19:43:58.971291', 72.0]], 'preserved_msgIds': [36125, 36130, 36406, 36450, 36486], 'prev_tempo': {'addominali': [[156.05405564900138, 111.02994525800023, 99.94005111100341, 97.61069813799986, 97.9964455670015, 91.4806724730006, 105.48080302200106, 107.126901636002, 125.81684930399933, 80.01495901100134, 80.01495901100134], [1.3476661240056274, 9.221072176995222, 112.67035331399529, 107.0965926219942, 140.38410791000206, 91.74315100700187, 76.9428340010054, 86.79510790500353, 8.745087525996496, 86.42529395500605, 86.42529395500605], [100, 100, 100, 100, 85, 85, 85, 85, 70, 70, 70]], 'pettobi': [[456.10524176499894, 127.79336847100058, 122.35872185799963, 132.28754203299468, 147.63920145599695, 161.25068290800118, 129.06809857599728, 112.81179034700472, 131.97456286700617, 119.28798122199805, 103.0892560909997, 235.1453804970006, 139.95653933100402, 108.76585689099738, 119.87495730200317, 152.14284440499614, 122.09336197198718, 150.29440029599937, 130.24911408800108, 124.33360110099602, 126.63049956699251, 169.92522055598965, 147.38983448999352, 137.62194637001085, 137.30152714699216, 137.9003726439987, 103.79556352099462, 103.79556352099462], [120, 120, 120, 120, 120, 85, 85, 85, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100], [120, 120, 120, 120, 120, 85, 85, 85, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100]], 'schiena': [[42.015306240005884, 85.75019623599655, 91.98791161400004, 82.8455477450043, 285.3548354779996, 81.3753198580016, 131.81698198999948, 140.6067977929997, 138.3048929200013, 156.83368331399834, 155.82047453699488, 226.18813510699692, 144.87260614000115, 144.2630102980038, 166.81588091800222, 289.37823802800267, 186.508419951002, 198.28363388300204, 262.6716593559977, 138.36426698500145, 138.36426698500145], [66.4217054159999, 95.77462775499953, 90.79615308199936, 103.58599200900062, 301.89677512600065, 139.88165118100005, 148.89739370000098, 129.66158106199873, 136.42410586799997, 263.640988915, 180.0425736039997, 194.78876624400073, 118.48784315400007, 152.84040112399998, 144.60361823899984, 200.59278008600086, 186.69544120599858, 174.27646199799892, 260.0723366080001, 120.81669442099883, 120.81669442099883], [100, 100, 100, 100, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120]], 'spalletri': [[180.1448306099992, 83.7315916439984, 85.42463757400037, 86.7639201940001, 90.68764061000184, 291.47758539100323, 11.297743769002409, 148.08050215500043, 133.9215232020033, 118.19201238900132, 161.45808495700112, 108.85746658500284, 109.39056832200004, 133.10755225499815, 139.76460772399878, 110.35530535900034, 120.04980469300062, 249.4628320970005, 198.74998035599856, 118.34003289799875, 135.9024515400015, 134.88953536899862, 287.2228129690011, 99.70586600000024, 94.9509411830004, 94.9509411830004], [5.504287674004445, 96.42483057100617, 96.22209244199621, 96.81179361999966, 86.52368478999415, 106.5325201079977, 158.69735562300775, 131.00018851298955, 121.5029678150022, 117.26818278399878, 127.83320070699847, 111.93345461699937, 114.4314488000091, 117.88109284300299, 113.12073416999192, 115.5502365949942, 117.78472933599551, 226.56726013100706, 165.65748278099636, 132.3947183470009, 134.64073086999997, 132.45129953000287, 97.23156634399493, 97.3312075290014, 94.67455334600527, 94.67455334600527], [100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100, 100]], 'altro': [[], [], []]}, 'test_dict': {'a': {'b': [[1, 2, 3], [1, 2, 3]]}}, 'tracebacks': {}, 'ultima_stat': {'y_aumenti_peso': 0, 'y_allenamenti': 0, 'y_addominali': 0}, 'ultimi_allenamenti': [["*Schiena*, `{'gio'} 23 Feb `", '*Addominali*, `mar 28 Feb `', '*Schiena*, `gio 02 Mar `'], ['2023-02-23T23:14:16.755518', '2023-02-28T22:23:31.470255', '2023-03-02T22:44:42.104510']], 'ultimo_allenamento': ['spalletri', 'addominali'], 'workout_count': {'pettobi': 0, 'spalletri': 0, 'addominali': 0, 'schiena': 0, 'altro': 0}}
#
# """
#
#
# string_list = []
# new_string = ""
# keysVals_list = string.split()
# for i, key in enumerate(db_keys):
#     if i == len(db_keys)-1:
#         new_string += split_string[-1]
#         break
#
#     key_string = f"'{db_keys[i+1]}': "
#     split_string = string.split(key_string)
#     new_string += split_string[0] + "\n\n" + key_string
#     string = split_string[-1]
#
#
# print(new_string)
#
# """
# {'allenamenti': [ObservedList(value=['2021-05-01T00:00:00', 'pettobi']), ObservedList(value=['2021-06-02T00:00:00', 'pettobi']), ObservedList(value=['2022-06-28T22:03:00.349815','pettobi']), ObservedList(value=['2022-06-30T22:02:27.841406', 'spalletri']), ObservedList(value=['2022-07-02T22:06:41.745507', 'pettobi', 'addominali']), ObservedList(value=['2022-07-04T22:04:21.529954', 'schiena']), ObservedList(value=['2022-07-06T16:10:49.662584', 'pettobi', 'addominali']), ObservedList(value=['2022-07-28T22:05:16.593999', 'spalletri', 'pettobi']), ObservedList(value=['2022-08-02T22:00:10.786688', 'addominali']), ObservedList(value=['2022-08-04T22:07:50.956170', 'schiena']), ObservedList(value=['2022-08-06T22:01:29.117742', 'spalletri', 'addominali']), ObservedList(value=['2022-08-08T22:07:18.017165', 'pettobi']), ObservedList(value=['2022-08-09T22:09:14.057384', 'schiena', 'addominali']), ObservedList(value=['2022-08-10T22:06:10.398188', 'spalletri']), ObservedList(value=['2022-08-12T22:02:55.719616', 'addominali', 'pettobi']), ObservedList(value=['2022-08-17T22:05:24.007626', 'schiena']), ObservedList(value=['2022-08-18T22:04:49.278957', 'spalletri']), ObservedList(value=['2022-08-19T22:01:25.444301', 'addominali']), ObservedList(value=['2022-08-22T22:06:57.185344', 'pettobi', 'addominali']), ObservedList(value=['2022-08-25T22:09:46.849795', 'schiena', 'addominali']), ObservedList(value=['2022-08-26T22:04:57.266191', 'spalletri']), ObservedList(value=['2022-08-29T22:05:55.810576', 'pettobi', 'addominali']), ObservedList(value=['2022-08-30T22:07:04.340869', 'schiena']), ObservedList(value=['2022-08-31T22:06:21.858619', 'spalletri']), ObservedList(value=['2022-09-02T22:07:08.919972', 'addominali']), ObservedList(value=['2022-10-01T22:04:55.202453', 'spalletri', 'addominali']), ObservedList(value=['2022-10-04T22:00:05.545632', 'pettobi', 'addominali']), ObservedList(value=['2022-10-07T22:02:02.631164', 'schiena', 'addominali']), ObservedList(value=['2022-10-08T22:09:43.564880', 'spalletri']), ObservedList(value=['2022-10-10T22:09:54.371073', 'addominali', 'pettobi']), ObservedList(value=['2022-10-12T22:03:04.374646', 'schiena']), ObservedList(value=['2022-10-14T22:07:21.647547', 'spalletri', 'addominali']), ObservedList(value=['2022-10-17T22:09:19.897295', 'pettobi', 'addominali']), ObservedList(value=['2022-10-20T22:04:01.201494', 'schiena', 'addominali']), ObservedList(value=['2022-10-21T22:00:50.981645', 'spalletri']), ObservedList(value=['2022-10-24T22:03:29.088096', 'pettobi', 'addominali']), ObservedList(value=['2022-11-04T23:06:03.019767', 'spalletri']), ObservedList(value=['2022-12-07T19:40:45.994393', 'addominali', 'pettobi']), ObservedList(value=['2022-12-20T23:05:33.345562', 'schiena']), ObservedList(value=['2022-12-27T23:01:32.130200', 'pettobi', 'addominali']), ObservedList(value=['2022-12-29T23:06:19.523063', 'schiena']), ObservedList(value=['2022-12-31T23:04:35.835921', 'spalletri', 'addominali']), ObservedList(value=['2023-01-02T23:06:53.067975', 'pettobi']), ObservedList(value=['2023-01-03T23:04:58.327716', 'schiena', 'addominali']), ObservedList(value=['2023-02-20T22:22:25.525690', 'spalletri']), ObservedList(value=['2023-02-22T22:04:40.901145', 'pettobi', 'addominali']), ObservedList(value=['2023-02-23T23:14:16.755518', 'schiena']), ObservedList(value=['2023-02-28T22:23:31.470255', 'addominali']), ObservedList(value=['2023-03-02T22:44:42.104510', 'schiena'])],
#
# 'aumenti_peso': [ObservedList(value=['2021-05-01T00:00:00', '34k', '36k', 'Crunch']), ObservedList(value=['2021-06-02T00:00:00', '34k', '36k', 'Crunch']), ObservedList(value=['2022-07-04T17:00:11.704611', '25㎏', 'mezzi stacchi', '30㎏']), ObservedList(value=['2022-07-04T17:31:13.740570', '25㎏ \+ 2 \(\- 2\) …', 'lat pull maniglie', '32㎏ + 2 (- 2) …']), ObservedList(value=['2022-07-08T16:44:13.923236', '10㎏', 'lento avanti', '10㎏ 2,5㎏']), ObservedList(value=['2022-07-11T17:35:58.054621', '32㎏; 18㎏', 'Curl BD', '36㎏; 14㎏']), ObservedList(value=['2022-08-08T11:41:33.587848', '6㎏', 'Curl spider 30°', '7㎏']), ObservedList(value=['2022-08-09T19:47:44.876065', '32㎏ \+ 2 \(\- 2\) …', 'lat pull maniglie', '25㎏ + 2 (- 2) …']), ObservedList(value=['2022-08-18T12:35:05.151178', '9㎏ \+ 2 \(\- 2\) …', 'shoulder press', '9㎏ + 2 (- 2) …']), ObservedList(value=['2022-09-02T18:08:27.981268', '18㎏', 'Spinte p 30°', '20㎏']), ObservedList(value=['2022-09-03T11:40:43.422265', '23㎏ \- 18㎏ \- 14㎏', 'rem cavo basso corda', '27㎏ - 23㎏ - 18㎏']), ObservedList(value=['2022-10-01T12:05:07.305885', '10㎏ 2,5㎏', '15㎏', 'Lento avanti']), ObservedList(value=['2022-10-04T18:31:28.617643', '25㎏ \+ 2', '32㎏ + 1', 'Chest press']), ObservedList(value=['2022-10-07T17:26:17.375654', '27㎏ \- 23㎏ \- 18㎏', '32㎏ - 27㎏ - 23㎏', 'Rem cavo basso corda']), ObservedList(value=['2022-10-10T17:09:36.769367', '16㎏', '18㎏', 'Spinte pp']), ObservedList(value=['2022-10-14T17:10:16.045986', '18㎏; 9㎏', '23㎏: 9㎏', 'Alzate front bd']), ObservedList(value=['2022-10-14T17:29:59.446353', '6㎏', '7㎏', 'Alzate lat manubri']), ObservedList(value=['2022-10-17T17:38:49.432628', '18㎏; 14㎏ \- …', '18㎏; 18㎏ - ...', 'Croci ai cavi']), ObservedList(value=['2022-10-17T17:48:00.538771', '20㎏', '22㎏', 'Spinte p 30°']), ObservedList(value=['2022-10-20T17:48:10.045238', '14㎏ \+ 2', '18㎏ + 2', 'Ss diverging seat row: orizz \| vertic']), ObservedList(value=['2022-10-20T17:48:23.734082', '25㎏ \+ 2 \(\- 2\) …', '32㎏ + 2 (- 2) …', 'Lat pull maniglie']), ObservedList(value=['2022-10-21T17:37:09.588936', '9㎏ \+ 2 \(\- 2\) …', '9㎏ + 2 (- 1) …', 'Shoulder press']), ObservedList(value=['2022-11-04T18:21:55.749141', '7㎏', '8㎏', 'Alzate laterali']), ObservedList(value=['2022-11-07T18:32:15.220871', '16㎏, 12㎏, 8㎏', '16㎏, 10㎏, 6㎏', 'Distensioni man\. p:1']), ObservedList(value=['2022-12-20T19:34:50.953011', '18㎏ \+ 1, 23㎏', '18㎏ + 1, 27㎏', 'Lat pull / pull down / lat pull']), ObservedList(value=['2022-12-23T18:09:55.695725', '16㎏ / 6㎏', '18㎏ / 6㎏', 'Lento man\. / arnold press']), ObservedList(value=['2022-12-23T19:10:23.128224', '5㎏ \(\+ 1,5㎏\)', '7,5㎏ (+ 1,25㎏)', 'French press ez']), ObservedList(value=['2022-12-29T18:11:13.082495', '18㎏ \+ 1, 27㎏', '18㎏ + 2, 27㎏', 'Lat pull / pull down / lat pull']), ObservedList(value=['2023-02-20T19:35:03.746611', '14㎏, 9㎏', '18㎏, 9㎏', 'Alzate frontali \(corda\)']), ObservedList(value=['2023-02-20T20:02:54.721372', '41㎏; 36㎏, 45㎏', '32㎏, 27㎏, 36㎏', 'Push down bd']), ObservedList(value=['2023-02-20T20:03:35.609091', '41㎏; 36㎏, 45㎏', '32㎏; 27㎏, 36㎏', 'Push down bd']), ObservedList(value=['2023-02-22T17:19:31.603513', '26㎏', '28㎏', 'Distensioni man\. 45°']), ObservedList(value=['2023-02-22T17:42:20.390988', '12㎏', '14㎏', 'Croci pp']), ObservedList(value=['2023-02-22T17:52:49.530590', '14㎏', '16㎏', 'Croci panca declinata 30°']), ObservedList(value=['2023-02-24T17:39:57.428096', '8㎏', '10㎏', 'Alzate laterali']), ObservedList(value=['2023-02-24T18:08:26.556634', '4㎏', '5㎏', 'Alzate lat\. \+ front\. \(comb\)']), ObservedList(value=['2023-03-01T18:42:25.837480', '36㎏ \(\- 1\)', '36㎏ + 2', 'Low row \(pr\. p\.\)']), ObservedList(value=['2023-03-01T18:42:55.818805', '36㎏ \(\- 1\)', '36㎏ + 2 (- 1)', 'Low row \(pr\. p\.\)']), ObservedList(value=['2023-03-01T19:07:49.657814', '39㎏', '39㎏ + 2', 'Lat pull']), ObservedList(value=['2023-03-01T19:08:24.305976', '15㎏', '20㎏', 'Dante row']), ObservedList(value=['2023-03-06T17:43:25.358682', '8㎏', '10㎏', 'Alzate laterali']), ObservedList(value=['2023-03-06T18:02:26.532864', '14㎏, 9㎏', '18㎏, 14㎏', 'Alzate frontali \(corda\)']), ObservedList(value=['2023-03-06T18:16:36.805253', '36㎏ \+ 1 \(\-1\)', '41㎏ (-1)', 'Shoulder press'])],
#
# 'calendario': [2023, 2],
#
# 'Cycle_4settLever': True,
#
# 'Cycle_dailyLever': True,
#
# 'Cycle_monthLever': True,
#
# 'data_2volte': {'pettobi': None, 'spalletri': None, 'addominali': None, 'schiena': None, 'altro': None},
#
# 'data_cambioscheda': ['2022-11-03T21:51:40.118463', '2023-02-08T00:10:31.600859'],
#
# 'data_prossima_stat': '2023-03-06T00:00:00',
#
# 'data_prossima_stat_m': [2023, 2, 28],
#
# 'dizAll_2volte': {'pettobi': None, 'spalletri': None, 'addominali': None, 'schiena': None, 'altro': None},
#
# 'fine_allenamenti': {'addominali': ObservedList(value=['`6 Mar`
# Tempo impiegato: *18m 37s*
# Dalle 18:42 alle 19:01', '`28 Feb`
# Tempo impiegato: *00m 41s*
# Dalle 19:21 alle 19:22', '`22 Feb`
# Tempo impiegato: *22m 15s*
# Dalle 18:18 alle 18:40', '`3 Gen`
# Tempo impiegato: *25m 31s*
# Dalle 17:48 alle 18:14']), 'pettobi': ObservedList(value=['`22 Feb`
# Tempo impiegato: *1h 07m 40s*
# Dalle 17:06 alle 18:14', '`2 Gen`
# Tempo impiegato: *1h 09m 59s*
# Dalle 17:28 alle 18:38', '`27 Dic`
# Tempo impiegato: *1h 14m 02s*
# Dalle 17:48 alle 19:02', '`7 Nov`
# Tempo impiegato: *N/A*
# Fine: 19:23']), 'schiena': ObservedList(value=['`1 Mar`
# Tempo impiegato: *55m 29s*
# Dalle 18:19 alle 19:15', '`23 Feb`
# Tempo impiegato: *55m 04s*
# Dalle 17:28 alle 18:23', '`3 Gen`
# Tempo impiegato: *52m 55s*
# Dalle 16:50 alle 17:43', '`29 Dic`
# Tempo impiegato: *47m 20s*
# Dalle 17:10 alle 17:58']), 'spalletri': ObservedList(value=['`6 Mar`
# Tempo impiegato: *58m 26s*
# Dalle 17:40 alle 18:38', '`20 Feb`
# Tempo impiegato: *49m 11s*
# Dalle 19:16 alle 20:05', '`31 Dic`
# Tempo impiegato: *54m 18s*
# Dalle 12:19 alle 13:13', '`4 Nov`
# Tempo impiegato: *N/A*
# Fine: 18:06']), 'altro': ObservedList(value=['addominali, 28/6, 1ora', 'addominali, 28/6, 1ora', 'addominali, 28/6, 1ora', 'addominali, 28/6, 1ora'])},
#
# 'last_msg_id': 36520,
#
# 'latest_messageId': 35836,
#
# 'latest_msgId': 36001,
#
# 'lever_cambioscheda': False,
#
# 'lever_thread_4sett': True,
#
# 'lever_thread_mens': True,
#
# 'lever_ultimi_allenamenti': True,
#
# 'peso_bilancia': [ObservedList(value=['2023-01-02T19:15:59.882048', 70.85]), ObservedList(value=['2023-01-10T19:29:20.740950', 70.9]), ObservedList(value=['2023-01-17T20:06:44.899653', 70.65]), ObservedList(value=['2023-01-19T19:57:53.002134', 70.6]), ObservedList(value=['2023-01-24T19:52:14.165599', 71.2]), ObservedList(value=['2023-01-26T20:06:03.379746', 71.25]), ObservedList(value=['2023-01-31T19:58:35.247931', 70.4]), ObservedList(value=['2023-02-23T19:49:04.873234', 71.3]), ObservedList(value=['2023-02-24T19:56:24.236216', 72.1]), ObservedList(value=['2023-02-28T19:43:58.971291', 72.0])],
#
# 'preserved_msgIds': [36125, 36130, 36406, 36450, 36486],
#
# 'prev_tempo': {'addominali': ObservedList(value=[ObservedList(value=[156.05405564900138, 111.02994525800023, 99.94005111100341, 97.61069813799986, 97.9964455670015, 91.4806724730006, 105.48080302200106, 107.126901636002, 125.81684930399933, 80.01495901100134, 80.01495901100134]), ObservedList(value=[1.3476661240056274, 9.221072176995222, 112.67035331399529, 107.0965926219942, 140.38410791000206, 91.74315100700187, 76.9428340010054, 86.79510790500353, 8.745087525996496, 86.42529395500605, 86.42529395500605]), ObservedList(value=[100, 100, 100, 100, 85, 85, 85, 85, 70, 70, 70])]), 'pettobi': ObservedList(value=[ObservedList(value=[456.10524176499894, 127.79336847100058, 122.35872185799963, 132.28754203299468, 147.63920145599695, 161.25068290800118, 129.06809857599728, 112.81179034700472, 131.97456286700617, 119.28798122199805, 103.0892560909997, 235.1453804970006, 139.95653933100402, 108.76585689099738, 119.87495730200317, 152.14284440499614, 122.09336197198718, 150.29440029599937, 130.24911408800108, 124.33360110099602, 126.63049956699251, 169.92522055598965, 147.38983448999352, 137.62194637001085, 137.30152714699216, 137.9003726439987, 103.79556352099462, 103.79556352099462]), ObservedList(value=[120, 120, 120, 120, 120, 85, 85, 85, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100]), ObservedList(value=[120, 120, 120, 120, 120, 85, 85, 85, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100])]), 'schiena': ObservedList(value=[ObservedList(value=[42.015306240005884, 85.75019623599655, 91.98791161400004, 82.8455477450043, 285.3548354779996, 81.3753198580016, 131.81698198999948, 140.6067977929997, 138.3048929200013, 156.83368331399834, 155.82047453699488, 226.18813510699692, 144.87260614000115, 144.2630102980038, 166.81588091800222, 289.37823802800267, 186.508419951002, 198.28363388300204, 262.6716593559977, 138.36426698500145, 138.36426698500145]), ObservedList(value=[66.4217054159999, 95.77462775499953, 90.79615308199936, 103.58599200900062, 301.89677512600065, 139.88165118100005, 148.89739370000098, 129.66158106199873, 136.42410586799997, 263.640988915, 180.0425736039997, 194.78876624400073, 118.48784315400007, 152.84040112399998, 144.60361823899984, 200.59278008600086, 186.69544120599858, 174.27646199799892, 260.0723366080001, 120.81669442099883, 120.81669442099883]), ObservedList(value=[100, 100, 100, 100, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120])]), 'spalletri': ObservedList(value=[ObservedList(value=[180.1448306099992, 83.7315916439984, 85.42463757400037, 86.7639201940001, 90.68764061000184, 291.47758539100323, 11.297743769002409, 148.08050215500043, 133.9215232020033, 118.19201238900132, 161.45808495700112, 108.85746658500284, 109.39056832200004, 133.10755225499815, 139.76460772399878, 110.35530535900034, 120.04980469300062, 249.4628320970005, 198.74998035599856, 118.34003289799875, 135.9024515400015, 134.88953536899862, 287.2228129690011, 99.70586600000024, 94.9509411830004, 94.9509411830004]), ObservedList(value=[5.504287674004445, 96.42483057100617, 96.22209244199621, 96.81179361999966, 86.52368478999415, 106.5325201079977, 158.69735562300775, 131.00018851298955, 121.5029678150022, 117.26818278399878, 127.83320070699847, 111.93345461699937, 114.4314488000091, 117.88109284300299, 113.12073416999192, 115.5502365949942, 117.78472933599551, 226.56726013100706, 165.65748278099636, 132.3947183470009, 134.64073086999997, 132.45129953000287, 97.23156634399493, 97.3312075290014, 94.67455334600527, 94.67455334600527]), ObservedList(value=[100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100, 100, 100, 100, 100, 100, 120, 120, 120, 120, 100, 100, 100, 100])]), 'altro': ObservedList(value=[ObservedList(value=[]), ObservedList(value=[]), ObservedList(value=[])])},
#
# 'tracebacks': {},
#
# 'ultima_stat': {'y_aumenti_peso': 0, 'y_allenamenti': 0, 'y_addominali': 0},
#
# 'ultimi_allenamenti': [ObservedList(value=[*Schiena*, `{'gio'} 23 Feb `, '*Addominali*, `mar 28 Feb `', '*Schiena*, `gio 02 Mar `']), ObservedList(value=['2023-02-23T23:14:16.755518', '2023-02-28T22:23:31.470255', '2023-03-02T22:44:42.104510'])],
#
# 'ultimo_allenamento': ['spalletri', 'addominali'],
#
# 'workout_count': {'pettobi': 0, 'spalletri': 0, 'addominali': 0, 'schiena': 0, 'altro': 0}}
# """
#
#
#
# def stringa_to_kg(stringa):
#     """
#     36k + 1 (-1), 27k + 2
#     split
#     [36k, +, 1, (-, 1), 27k, +, 2]
#     if ( in split remove
#     [36k, +, 1, 27k, +, 2]
#     if [1,2,3,4,5,6,7,8,9] not in split
#     [36k, 1, 27k, 2]
#     count k : 2  # divideremo il numero finale per questo numero
#     moltiplicazione numeri senza k: if k not in split n * 1.5
#     [36k, 1.5, 27k, 3]
#     rimozione k
#     [36, 1.5, 27, 3]
#     sum / 2
#     67.5 / 2
#     """
#     peso_disco_macchinario = 1.5  # rappresenta il peso di un disco extra di un macchinario in kg
#
#     stringa = stringa.replace(",", ".")  # sostituiamo le , coi . in modo da poter usare float()
#     split = stringa.split()
#
#     # prima vengono selezionati solo i pieces che non contengono parentesi e poi quelli che contengono numeri
#     keep_idxs = []
#     for piece_i, piece in enumerate(split):
#         if "(" not in piece and ")" not in piece:
#             if any(char.isdigit() for char in piece):
#                 keep_idxs.append(piece_i)
#
#     split = [split[idx] for idx in keep_idxs]
#
#     # se ha k l'idx viene aggiunto a k_idxs, altrimenti a num idxs
#     k_idxs = []
#     num_idxs = []
#     [k_idxs.append(idx) if "k" in piece else num_idxs.append(idx) for idx, piece in enumerate(split)]
#
#     division_factor = max(1, len(k_idxs))  # numero per cui dividiamo la somma finale per ottenere la media, se non ci sono k_idxs è 1
#
#     for num_idx in num_idxs:
#         split[num_idx] = float(split[num_idx]) * peso_disco_macchinario
#
#     for k_idx in k_idxs:
#         split[k_idx] = float(split[k_idx][:-1])
#
#     average = sum(split) / division_factor
#     return average
#
#
#
# a = stringa_to_kg("dscsd")
# ic(a)