#coding=utf-8
from os.path import join, isfile, isdir
from os import listdir
import numpy as np
import argparse
import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import sys



def main():
    parser = argparse.ArgumentParser(description="Error predictor network",
                                     epilog="v0.0.1")
    parser.add_argument("job_folder",
                        action="store",
                        help="path to job foder")

    parser.add_argument("--gpu",
                        "-gpu",
                        action="store_true",
                        default=False,
                        help="select if you want to predit QA on GPU, default on CPU (optional)")

    args = parser.parse_args()

    logo()


    # ========================== Defining parameters ==========================
    job_folder = args.job_folder
    job_status = join(job_folder,"status.txt")

    job_infos = open(join(job_folder,"info.txt"), 'r')
    user_jobName = job_infos.readline().replace('\n', '').split(" ")[-1]
    user_email = job_infos.readline().replace('\n', '').split(" ")[-1]

    print("===============================================")
    print("============== User Information ===============")
    print("   user_email:   ", user_email)
    print("   user_jobName: ", user_jobName)
    print("===============================================\n")


    #========================== Extracting input parameters ==========================
    if args.gpu:
        device = ""
    else:
        device = " -cpu"

    model_Glddt = []

    print("========================== Folding initial model ==========================")
    cmd = "echo fold_init_model >> " + job_status
    print(cmd)
    os.system(cmd)

    fasta = join(job_folder,"seq.fasta")
    msa = join(job_folder,"msa.a3m")
    msa_feat_fold = join(job_folder, "msa.fold.npz")
    geometries = join(job_folder,"geometries-init.npz")
    init_model = join(job_folder,"model-1.pdb")

    search_msa(fasta, job_folder, msa)

    if user_email[:6] == "sd-m__":
        msa_feat = join(job_folder, "msa.npz")
        extract_msa_feat(msa, msa_feat)

        ss = join(job_folder, "msa.ss")
        msa_ss = join(job_folder, "msa.ss_a3m")
        templ = join(job_folder, "templ")
        templ_feat = join(job_folder, "templ.npz")
        search_template(msa, ss, msa_ss, templ)
        extract_template_feat(fasta, templ, templ_feat)

        cmd = "echo finish" + " >> " + job_status
        print(cmd)
        os.system(cmd)
        exit(0)


    extract_msa_feat_fold(msa, msa_feat_fold)
    folding_structure_by_RocketX(fasta, msa_feat_fold, geometries, init_model, device)


    print("========================== Preparing for model evaluation ==========================")
    cmd = "echo prepare_evlate >> " + job_status
    print(cmd)
    os.system(cmd)

    msa_feat = join(job_folder, "msa.npz")
    ss = join(job_folder, "msa.ss")
    msa_ss = join(job_folder, "msa.ss_a3m")
    templ = join(job_folder,"templ")
    templ_feat = join(job_folder, "templ.npz")
    search_template(msa, ss, msa_ss, templ)

    extract_msa_feat(msa, msa_feat)
    extract_template_feat(fasta, templ, templ_feat)


    print("========================== Evaluating initial model quality ==========================")
    cmd = "echo evaluate_init_model >> " + job_status
    print(cmd)
    os.system(cmd)
    model = join(job_folder, "model-1.pdb")
    model_feat = join(job_folder, "model-1.feat.npz")
    model_QA = join(job_folder, "model-1.QA.npz")
    model_G_lddt = join(job_folder, "model-1.global_lddt.txt")
    model_evaluated = join(job_folder, "model-1.evaluated.pdb")
    model_figure = join(job_folder, "model-1")

    if not isfile(model_QA):
        extract_model_feat(model, model_feat)
        predict_QA(model_feat, msa_feat, templ_feat, model_QA, device)
    else:
        print("   ", model_QA, "already exist!")
    gLDDT = global_lddt(model_QA, model_G_lddt)
    if not isfile(model_evaluated):
        QA2pdb(model_QA, model, model_evaluated)
    if not isfile(model_figure+".deviation.png") or not isfile(model_figure+".lddt.png"):
        plot_QA(model_QA, model_figure)

    model_Glddt.append([model[:-4], gLDDT])

    cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/renumber_pdb.py " + model + " tempfile.pdb"
    os.system(cmd)
    cmd = "mv -f tempfile.pdb " + model
    os.system(cmd)

    print("========================== Refinement models ==========================")
    for i in range(2, 6):
        model = join(job_folder, "model-" + str(i-1) + ".pdb")
        refined_model = join(job_folder, "model-" + str(i) + ".pdb")

        print("==================== Refining model " + str(i) + " ====================")
        cmd = "echo refine_model-" + str(i) +" >> " + job_status
        print(cmd)
        os.system(cmd)

        if not isfile(refined_model):
            cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/refine/refine2.py " + job_folder + " " + model[:-4] + " " + refined_model[:-4]
            print(cmd)
            os.system(cmd)
        else:
            print("   ", refined_model, "already exist!")

        refined_model_feat = join(job_folder, "model-" + str(i) + ".feat.npz")
        refined_model_QA = join(job_folder, "model-" + str(i) + ".QA.npz")
        refined_model_G_lddt = join(job_folder, "model-" + str(i) + ".global_lddt.txt")
        refined_model_evaluated = join(job_folder, "model-" + str(i) + ".evaluated.pdb")
        refined_model_figure = join(job_folder, "model-" + str(i))

        print("==================== Evaluating model " + str(i) + " ====================")
        cmd = "echo evaluate_model-" + str(i) + " >> " + job_status
        print(cmd)
        os.system(cmd)

        if not isfile(refined_model_QA):
            extract_model_feat(refined_model, refined_model_feat)
            predict_QA(refined_model_feat, msa_feat, templ_feat, refined_model_QA, device)
        else:
            print("   ", refined_model_QA, "already exist!")
        refined_gLDDT = global_lddt(refined_model_QA, refined_model_G_lddt)

        if not isfile(refined_model_evaluated):
            QA2pdb(refined_model_QA, refined_model, refined_model_evaluated)
        if not isfile(refined_model_figure + ".deviation.png") or not isfile(model_figure + ".lddt.png"):
            plot_QA(refined_model_QA, refined_model_figure)

        model_Glddt.append([refined_model[:-4], refined_gLDDT])

        cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/renumber_pdb.py " + refined_model_evaluated + " tempfile.pdb"
        os.system(cmd)
        cmd = "mv -f tempfile.pdb " + refined_model_evaluated
        os.system(cmd)

    print("========================== Refinement finish ==========================")
    model_Glddt.sort(key=lambda x: (x[1], x[0]), reverse=True)
    print(model_Glddt)

    results_path = join(job_folder, "results")
    cmd = "mkdir " + results_path
    os.system(cmd)
    for i in range(0, len(model_Glddt)):
        model = join(job_folder, model_Glddt[i][0] + ".pdb")
        result_model = join(results_path, "model" + str(i+1) + ".pdb")
        cmd = "cp " + model + " " + result_model
        os.system(cmd)

        if i == 0:
            fig_lddt = join(job_folder, model_Glddt[i][0] + ".lddt.png")
            fig_esto = join(job_folder, model_Glddt[i][0] + ".deviation.png")
            lddt_path = join(results_path, "lddt.png")
            esto_path = join(results_path, "deviation.png")

            cmd = "cp " + fig_lddt + " " + lddt_path
            os.system(cmd)
            cmd = "cp " + fig_esto + " " + esto_path
            os.system(cmd)
            cmd = "cp " + join(job_folder, model_Glddt[i][0] + ".QA.npz")  + " " + join(results_path, "model_evaluation.npz")
            os.system(cmd)

            QA_feat = join(job_folder, model_Glddt[i][0] + ".QA.npz")
            final_geometries = join(job_folder, "geometries-final.npz")
            predicting_final_geometries(msa_feat_fold, QA_feat, final_geometries, device)
            plot_Geom(final_geometries, results_path)

            cmd = "cp " + final_geometries  + " " + join(results_path, "geometries.npz")
            os.system(cmd)

    # success = False
    if isfile(join(results_path, "model1.pdb")) and isfile(join(results_path, "model2.pdb")) and isfile(join(results_path, "model3.pdb")) \
            and isfile(join(results_path, "model4.pdb")) and isfile(join(results_path, "model5.pdb")) and isfile(join(results_path, "lddt.png")) \
            and isfile(join(results_path, "deviation.png")) and isfile(join(results_path, "dist.png")) and isfile(join(results_path, "omega.png")) \
            and isfile(join(results_path, "theta.png")) and isfile(join(results_path, "phi.png")) :
        success = True
    else:
        success = False

    print(success)
    if success:
        cmd = "echo finish" + " >> " + job_status
        print(cmd)
        os.system(cmd)
    else:
        cmd = "echo error" + " >> " + job_status
        print(cmd)
        os.system(cmd)








def search_msa(fasta, job_folder, msa):
    print("======== Searching MSA ...")
    if not isfile(msa):
        cmd = "/home/data/user/junl/RocketX-v1.0/scripts/make_msa.sh " + fasta + " " + job_folder + " " + msa + " 20 1000000"
        print(cmd)
        os.system(cmd)
    else:
        print("   ", msa, " already exit!")


def extract_msa_feat_fold(msa, msa_feat):
    print("======== Extracting msa features ...")
    if not isfile(msa_feat):
        cmd = "python /home/data/user/junl/RocketX-v1.0/scripts/msa2featrue.py " + msa + " " + msa_feat
        print(cmd)
        os.system(cmd)
    else:
        print("   ", msa_feat, " already exit!")


def extract_msa_feat(msa, msa_feat):
    print("======== Extracting msa features ...")
    if not isfile(msa_feat):
        num_seq = 0
        for line in open(msa, encoding='utf8'):
            if len(line) != 0:  num_seq += 1
        num_seq = num_seq / 2
        print("   nubmer of homologous sequence: ", num_seq)

        if num_seq >= 1000:
            cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/feature_from_msa.py " + msa + " " + msa_feat
            print(cmd)
            os.system(cmd)
        else:
            print("   Warning: ", msa_feat, " not sufficient homologous sequence!")
    else:
        print("   ", msa_feat, " already exit!")


def folding_structure_by_RocketX(fasta, msa_feat, geometries, init_model, device):
    print("======== folding_structure_by_RocketX ...")
    if not isfile(geometries):
        cmd = "python /home/data/user/junl/RocketX-v1.0/GeomNet/geom_pred_round1.py " + msa_feat + " " + geometries + device
        print(cmd)
        os.system(cmd)
    else:
        print("   ", geometries, " already exit!")

    if not isfile(init_model):
        cmd = "python /home/data/user/junl/RocketX-v1.0/Folding/trRosetta.py " + geometries + " " + fasta + " " + init_model
        print(cmd)
        os.system(cmd)
    else:
        print("   ", init_model, " already exit!")

def predicting_final_geometries(msa_feat, QA_feat, final_geometries, device):
    print("======== predicting final geometries ...")
    if not isfile(final_geometries):
        cmd = "python /home/data/user/junl/RocketX-v1.0/GeomNet/geom_pred_round2.py " + msa_feat + " " + QA_feat + " " + final_geometries + device
        print(cmd)
        os.system(cmd)
    else:
        print("   ", final_geometries, " already exit!")

def search_template(msa, ss, msa_ss, templ):
    print("======== Searching template ...")
    if not isfile(templ + ".hhr"):
        print("   extracting Secondary Structure ...")
        if not isfile(msa_ss) or not isfile(msa_ss):
            cmd = "/home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/make_ss.sh " + msa + " " + ss
            print(cmd)
            os.system(cmd)
            cmd = "cat " + ss + " " + msa + " > " + msa_ss
            os.system(cmd)
        else:
            print("   ", msa_ss, " already exit!")

        cmd = "/home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/make_template.sh " + msa_ss + " " + templ + " 20"
        print(cmd)
        os.system(cmd)
    else:
        print("   ", templ + ".hhr", " already exit!")


def extract_template_feat(fasta, templ, templ_feat):
    """

    :rtype: object
    """
    print("======== Extracting template feautres ...")
    if not isfile(templ_feat):
        num_hhr_line = 0
        for line in open(templ + ".hhr", "r").readlines():
            num_hhr_line += 1
            if num_hhr_line >= int(10) and num_hhr_line <= int(34):
                num_templ = num_hhr_line - 9
                prob = float(line[35:40])
                eval = float(line[41:48])
                if eval > 0.001 or prob < 60.0:
                    num_templ -= 1
                    break
                if num_hhr_line == 29:
                    break
        print("   Number of good template: ", num_templ)
        if num_templ <= 0:
            print("   Warning: ", templ_feat, " no good templ!!!")
            return

        L = 0
        for line in open(fasta, "r").readlines():
            if line[0] == '>':
                continue
            L += line.strip().__len__()

        cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/feature_from_template.py " + fasta + " " + templ + ".hhr " + templ + ".atab " + templ_feat + " -n_templ " + str(num_templ)
        print(cmd)
        os.system(cmd)
        # template2feat(templ + ".hhr", templ + ".atab", templ_feat, num_templ, L)
    else:
        print("   ", templ_feat, " already exit!")


def extract_model_feat(in_pdb, model_feat):
    print("======== Extract features from input model ...")
    if not isfile(model_feat):
        # pdb2feat(in_pdb, model_feat)
        cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/feature_from_model.py " + in_pdb + " " + model_feat
        print(cmd)
        os.system(cmd)
    else:
        print("   ", model_feat, " already exit!")


def predict_QA(model_feat, msa_feat, templ_feat, QA, device):
    print("======== Predicting model quality ...")
    if isfile(msa_feat) and isfile(templ_feat):
        print("   Running DeepUMQA2-Model-MSA-Template ...")
        cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/predict_from_model-msa-template.py " + model_feat + " " + msa_feat + " " + templ_feat + " " + QA + " " + device
        print(cmd)
        os.system(cmd)
    elif isfile(msa_feat):
        print("   Running DeepUMQA2-Model-MSA ...")
        cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/predict_from_model-msa.py " + model_feat + " " + msa_feat + " " + QA + " " + device
        print(cmd)
        os.system(cmd)
    elif isfile(templ_feat):
        print("   Running DeepUMQA2-Model-Template ...")
        cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/predict_from_model-template.py " + model_feat + " " + templ_feat + " " + QA + " " + device
        print(cmd)
        os.system(cmd)
    else:
        print("   Running DeepUMQA2-Model ...")
        cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/predict_from_model.py " + model_feat + " " + QA + " " + device
        print(cmd)
        os.system(cmd)


def global_lddt(QA, G_lddt):
    lddt = np.load(QA, allow_pickle=True)['lddt']
    global_lddt = str(round(np.mean(lddt) * 100, 2)) + "\n"
    out_Glddt = open(G_lddt, 'w')
    out_Glddt.write(global_lddt)
    out_Glddt.close()
    return round(np.mean(lddt) * 100, 2)

def plot_QA(QA, job_folder):
    print("======== Ploting deviation and lddt ...")
    cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/plot_QA.py " + QA + " " + job_folder
    os.system(cmd)

def plot_Geom(QA, job_folder):
    print("======== Ploting geometries ...")
    cmd = "python /home/data/user/junl/DeepUMQA/DeepUMQA2/scripts/plot_geometries.py " + QA + " " + job_folder
    os.system(cmd)

def QA2pdb(QA, in_pdb, out_pdb):
    lddt = np.load(QA, allow_pickle=True)['lddt']

    res_score = []
    for score in lddt:
        score = round(score * 100, 2)
        res_score.append(score)

    out_pdb = open(out_pdb, 'w')
    flag = 0
    for line in open(in_pdb, "r").readlines():
        if line.startswith('ATOM'):
            atom = line[12:16].strip()
            if atom == "N":
                flag += 1
                tmp = str(res_score[flag - 1])
            newline = line[0:60] + '%6s' % tmp + line[66:]
            out_pdb.write(newline)
        else:
            out_pdb.write(line)


def logo():
    print('*********************************************************************')
    print('*********************************************************************')
    print('*                                                                   *')
    print('*                          DGMFold server                           *')
    print('*                                                                   *')
    print("*   Please email your comments to: guijunlab01@163.com              *")
    print('*********************************************************************')
    print('*********************************************************************')




if __name__ == "__main__":
    main()

