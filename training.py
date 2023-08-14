from funzioni import secs_to_hours
import tensorflow as tf
import time
import json
import numpy as np
import shutil


def train_model(folder, model_name, db):

    start = time.perf_counter()

    num_classes = 4
    tipi = ["nomi", "reps", "peso", "note"]
    new_samples = 0  # rimarrà 0 solo in caso effettivamente db["new_training_data"] è = a []

    # AGGIUNTA NUOVI SAMPLES A TRAINING DATA
    if db["new_training_data"] != []:  # in caso ci sia stato /forcetraining
        TD_filePath = f"{folder}/training_data.json"
        with open(TD_filePath, "r") as file:
            trainig_data_json = json.load(file)

        num_samples = dict(nomi=[], reps=[], peso=[], note=[])  # [0] samples prima di aggiunta di new_training_data, [1] samples dopo
        [num_samples[tipo].append(len(samples)) for tipo, samples in trainig_data_json.items()]

        for text, key in db["new_training_data"]:
            trainig_data_json[key].append(text)

        with open(TD_filePath, "w") as file:
            json.dump(trainig_data_json, file, indent=4)

        new_samples = len(db["new_training_data"])  # salviamo la lunghezza prima di resettare
        db["new_training_data"] = []

    else:
        num_samples = dict(nomi=[0], reps=[0], peso=[0], note=[0])  # [0] samples prima di aggiunta di new_training_data, [1] samples dopo

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

    training_ds, train_oneHots = get_dataset(f"{folder}/training_data.json")
    testing_ds, _ = get_dataset(f"{folder}/training_data.json")

    # TRAINING
    # creaizone class weights per training bilanciato
    from sklearn.utils.class_weight import compute_class_weight
    train_labels_idxs = [np.argmax(np.array(lista)) for lista in train_oneHots]  # tutti gli one hots sottoforma di indexes
    class_weights = compute_class_weight(class_weight="balanced", classes=list(range(num_classes)), y=train_labels_idxs)
    class_weights = {i: weight for i, weight in enumerate(class_weights)}

    n_epochs = 100
    lr = 0.0005
    label_smoothing = 0.2

    model = tf.keras.models.load_model(f"{folder}/{model_name}")
    model.compile(tf.optimizers.Adam(learning_rate=lr), tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), "accuracy")

    fit = model.fit(training_ds, epochs=n_epochs, class_weight=class_weights)
    saved_model_fpath = f"{folder}/{model_name}"
    model.save(saved_model_fpath, save_format="tf")
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

<b>Dataset</b>
    new samples: {new_samples}
    <code>E: {ns[E][-1]}, {ns_p[E]:.0%} ({ns_d[E]:+.0f})  R: {ns[R][-1]}, {ns_p[R]:.0%} ({ns_d[R]:+.0f})</code> 
    <code>P: {ns[P][-1]}, {ns_p[P]:.0%} ({ns_d[P]:+.0f})  N: {ns[N][-1]}, {ns_p[N]:.0%} ({ns_d[N]:+.0f})</code>

<b>Training</b>
    class weights: (<code>E: {class_weights[0]:.3f}, R: {class_weights[1]:.3f}, P: {class_weights[2]:.3f}, N: {class_weights[3]:.3f}</code>)
    epochs: {n_epochs}, lr: {lr}, label smoothing: {label_smoothing}

<b>Results</b>
    <code>train: A: {fit.history["accuracy"][-1]:.2%}            L: {fit.history["loss"][-1]:.3f}</code> 
    <code>test:  A: {eval[-1]:.2%} ({val_diff[-1]:+.2%})  L: {eval[0]:.3f} ({val_diff[0]:.3f})</code>
    ⚠️ prendere nota del fatto che il dataset di testing è abbastanza lontano da quello di training e non ha reps e peso.
    """

    # db
    db["val_results"] = [eval[0], eval[-1]]
    db["NNs_names"][model_name]["version"] = db["NNs_names"][model_name]["version"] + 1

    # invio files
    zipped_model_fpath = f"{folder}/{db['NNs_names'][model_name]['name']} V{db['NNs_names'][model_name]['version']}"

    shutil.make_archive(zipped_model_fpath, "zip", saved_model_fpath)
    loaded_files = []
    for file in [f"{zipped_model_fpath}.zip", f"{folder}/training_data.json", f"{folder}/testing_data.json"]:
        loaded_files.append(open(file, "rb"))


    del model  # risparmio di memoria

    return summary_msg, loaded_files

