
"""
Steering file for reconstructing B -> K* ell+ ell-.


Submit to the Grid like this:

gbasf2 \
    -p mu_sideb_generic_mix \
    -s light-2401-ocicat \
    -i /belle/MC/release-06-00-08/DB00002100/MC15ri_b/prod00024821/s00/e1003/4S/r00000/mixed/mdst \
    /home/belle2/elee20/ml-hep-proj/experiments/2024-06-04_eff_res/scripts/recon_steer.py


Dataset paths:

    mixed bkg first file: -i /belle/MC/release-06-00-08/DB00002100/MC15ri_b/prod00024821/s00/e1003/4S/r00000/mixed/mdst \
    mixed bkg mini list: --input_dslist /home/belle2/elee20/ml-hep-proj/scripts/reconstruct/lpns_results_gen_mixed_mini.txt \
    mixed bkg all list: --input_dslist /home/belle2/elee20/ml-hep-proj/scripts/reconstruct/lpns_results_gen_mixed.txt \
    charged bkg all list: --input_dslist /home/belle2/elee20/ml-hep-proj/scripts/reconstruct/lpns_results_gen_charged.txt \
    signal mu: /belle/MC/release-06-00-08/DB00002100/MC15ri_b/prod00025631/s00/e1003/4S/r00000/1120240011/mdst
    signal e: /belle/MC/release-06-00-08/DB00002100/MC15ri_b/prod00025630/s00/e1003/4S/r00000/1120240010/mdst
    signal e first: -i /belle/MC/release-06-00-08/DB00002100/MC15ri_b/prod00025630/s00/e1003/4S/r00000/1120240010/mdst/sub00/mdst_000001_prod00025630_task10020000001.root \
"""


import basf2 as b2
import modularAnalysis as ma
from variables import collections as vc
from variables import utils as vu
from variables import variables as vm
import vertex as vx


# Use command line arguments here when in automated pipeline (sys.argv)
ell = "mu"  # 'mu' or 'e'"
input_filepaths = ""  # overwritten on the Grid
output_filepath = f"{ell}_recon.root"
sideband = True


main = b2.Path()


def append_global_tag():
    gt = ma.getAnalysisGlobaltag()
    b2.conditions.append_globaltag(gt)


def define_aliases():
    vm.addAlias("goodFWDGamma", "passesCut(clusterReg == 1 and clusterE > 0.075)")
    vm.addAlias("goodBRLGamma", "passesCut(clusterReg == 2 and clusterE > 0.05)")
    vm.addAlias("goodBWDGamma", "passesCut(clusterReg == 3 and clusterE > 0.1)")
    vm.addAlias("goodGamma", "passesCut(goodFWDGamma or goodBRLGamma or goodBWDGamma)")

    vm.addAlias("invM_Kst", "0.892")
    vm.addAlias("fullwidth_Kst", "0.05")

    vm.addAlias('tfChiSq', 'extraInfo(chiSquared)')
    vm.addAlias('tfNdf', 'extraInfo(ndf)')
    vm.addAlias('tfRedChiSq', 'formula(tfChiSq / tfNdf)')
    vm.addAlias('tfRedChiSqB0', 'extraInfo(tfRedChiSqB0)')

    vm.addAlias('CMS3_weMissM2', 'weMissM2(my_mask,3)')

    vm.addAlias('mcMother_mcPDG', 'mcMother(mcPDG)')
    vm.addAlias('mcSister_0_mcPDG', 'mcMother(mcDaughter(0, mcPDG))')
    vm.addAlias('mcSister_1_mcPDG', 'mcMother(mcDaughter(1, mcPDG))')
    vm.addAlias('mcSister_2_mcPDG', 'mcMother(mcDaughter(2, mcPDG))')
    vm.addAlias('mcSister_3_mcPDG', 'mcMother(mcDaughter(3, mcPDG))')
    vm.addAlias('mcSister_4_mcPDG', 'mcMother(mcDaughter(4, mcPDG))')
    vm.addAlias('mcSister_5_mcPDG', 'mcMother(mcDaughter(5, mcPDG))')
    vm.addAlias('mcSister_6_mcPDG', 'mcMother(mcDaughter(6, mcPDG))')
    vm.addAlias('mcSister_7_mcPDG', 'mcMother(mcDaughter(7, mcPDG))')
    vm.addAlias('mcSister_8_mcPDG', 'mcMother(mcDaughter(8, mcPDG))')
    vm.addAlias('mcSister_9_mcPDG', 'mcMother(mcDaughter(9, mcPDG))')    

    vm.addAlias('e_id_BDT', 'pidChargedBDTScore(11, ALL)')


def input_to_the_path():

    ma.inputMdstList(
        filelist=input_filepaths,
        path=main,
        environmentType="default",
    )


def reconstruct_generator_level():

    ma.fillParticleListFromMC(decayString="K+:gen", cut="", path=main)
    ma.fillParticleListFromMC(decayString="pi-:gen", cut="", path=main)
    ma.fillParticleListFromMC(decayString=f"{ell}+:gen", cut="", path=main)

    ma.reconstructMCDecay("K*0:gen =direct=> K+:gen pi-:gen", cut="", path=main)
    ma.reconstructMCDecay(f"B0:gen =direct=> K*0:gen {ell}+:gen {ell}-:gen", cut="", path=main)


def reconstruct_detector_level():

    marker = ''
    if ell == 'e':
        marker = '?addbrems'

    dz_cut = "[abs(dz) < 4]"
    dr_cut = "[dr < 2]"

    muonID_cut = "[muonID > 0.8]"
    muon_p_cut = "[p > 0.6]" 

    electronID_cut = "[e_id_BDT > 0.8]"
    electron_p_cut = "[p > 0.2]"

    kaonID_cut = "[kaonID > 0.8]"

    invMKst_cut = "[abs(formula(daughterInvM(0, 1) - invM_Kst)) <= formula(2 * fullwidth_Kst)]"

    deltaE_cut = '[abs(deltaE) <= 0.05]'    
    if ell == 'e':
        deltaE_cut = '[-0.075 <= deltaE <= 0.05]'

    Mbc_cut = "[Mbc > 5.27]"
    if sideband:
        Mbc_cut = "[5.2 < Mbc < 5.26]"
        
    muon_cut = f"{muonID_cut} and {muon_p_cut} and {dr_cut} and {dz_cut}  and thetaInCDCAcceptance"

    electron_cut = f"{electronID_cut} and {electron_p_cut} and {dr_cut} and {dz_cut} and thetaInCDCAcceptance"

    kaon_cut = f"{kaonID_cut} and {dr_cut} and {dz_cut} and thetaInCDCAcceptance"

    pion_cut = f"{dr_cut} and {dz_cut} and thetaInCDCAcceptance"

    if ell == 'e':
        ma.fillParticleList(decayString="e+:raw", cut='', path=main)

        ma.fillParticleList(decayString="gamma:brems", cut="goodGamma", path=main)
        ma.correctBrems("e+:det", "e+:raw", "gamma:brems", path=main)

        ma.applyChargedPidMVA(['e+:det'], path=main, trainingMode=1)
        ma.applyCuts("e+:det", electron_cut, path=main)

    elif ell == 'mu':
        ma.fillParticleList(decayString="mu+:det", cut=muon_cut, path=main)

    ma.fillParticleList(decayString="K+:det", cut=kaon_cut, path=main)
    ma.fillParticleList(decayString="pi-:det", cut=pion_cut, path=main)

    ma.reconstructDecay(
        "K*0:det =direct=> K+:det pi-:det", 
        cut=f"{invMKst_cut}", 
        path=main
    )

    ma.reconstructDecay(
        f"B0:det =direct=> K*0:det {ell}+:det {ell}-:det {marker}", 
        cut=f"{deltaE_cut} and {Mbc_cut}", 
        path=main
    )

    ma.matchMCTruth("B0:det", path=main)


def treefit():
    vx.treeFit('B0:det', conf_level=0.00, updateAllDaughters=True, ipConstraint=True, path=main)
    ma.variablesToExtraInfo('B0:det', {'tfRedChiSq':'tfRedChiSqB0'}, option=0, path=main)


def rest_of_event():
    # build the ROE
    ma.buildRestOfEvent('B0:det', fillWithMostLikely=True, path=main)

    loose_track = 'dr<10 and abs(dz)<20 and thetaInCDCAcceptance and E<5.5' 
    loose_gamma = "0.05 < clusterE < 5.5"
    tight_track = f'nCDCHits>=0 and thetaInCDCAcceptance and pValue>=0.0005 and \
                    [pt<0.15 and formula(dr**2/36+dz**2/16)<1] or \
                    [0.15<pt<0.25 and formula(dr**2/49+dz**2/64)<1] or \
                    [0.25<pt<0.5 and formula(dr**2/49+dz**2/16)<1] or \
                    [0.5<pt<1 and formula(dr**2/25+dz**2/36)<1] or \
                    [pt>1 and formula(dr**2+dz**2)<1]'
    tight_gamma = f'clusterE>0.05 and abs(clusterTiming)<formula(2*clusterErrorTiming) and abs(clusterTiming)<200'
    roe_mask1 = ('my_mask',  loose_track, loose_gamma)
    ma.appendROEMasks('B0:det', [roe_mask1], path=main)

    # creates V0 particle lists and uses V0 candidates to update/optimize the Rest Of Event
    ma.updateROEUsingV0Lists('B0:det', mask_names='my_mask', default_cleanup=True, selection_cuts=None,
                            apply_mass_fit=True, fitter='treefit', path=main)
    ma.updateROEMask("B0:det", "my_mask", tight_track, tight_gamma, path=main)


def printMCParticles():
    ma.printMCParticles(onlyPrimaries=False, suppressPrint=True, path=main)


def create_variable_lists():
    std_vars = (
        vc.deltae_mbc
        + vc.inv_mass
        + vc.mc_truth
        + ['mcMother_mcPDG']
        + ['PDG']
        + ['mcSister_0_mcPDG', 'mcSister_1_mcPDG', 'mcSister_2_mcPDG', 'mcSister_3_mcPDG', 'mcSister_4_mcPDG', 'mcSister_5_mcPDG', 'mcSister_6_mcPDG', 'mcSister_7_mcPDG', 'mcSister_8_mcPDG', 'mcSister_9_mcPDG']
        + vc.pid
        + vc.kinematics
        + vc.mc_kinematics
        + ['dr', 'dz']
        + ['theta', 'thetaErr', 'mcTheta']
        + ['isSignalAcceptBremsPhotons']
        + ['CMS3_weMissM2']
    )

    Kstar0_vars = vu.create_aliases_for_selected(
        list_of_variables=std_vars,
        decay_string=f"B0 -> [^K*0 -> K+ pi-] {ell}+ {ell}-",
    )

    K_pi_vars = vu.create_aliases_for_selected(
        list_of_variables=std_vars,
        decay_string=f"B0 -> [K*0 -> ^K+ ^pi-] {ell}+ {ell}-",
        prefix=["K_p", "pi_m"],
    )

    lepton_vars = vu.create_aliases_for_selected(
        list_of_variables=std_vars + ['e_id_BDT'],
        decay_string=f"B0 -> [K*0 -> K+ pi-] ^{ell}+ ^{ell}-",
        prefix=[f"{ell}_p", f"{ell}_m"],
    )

    B0_vars = (
        std_vars
        + ['tfRedChiSqB0']
        + Kstar0_vars
        + K_pi_vars 
        + lepton_vars
    )

    return B0_vars


def save_output(B0_vars):
    ma.variablesToNtuple(
        decayString="B0:gen",
        variables=B0_vars,
        filename=output_filepath,
        treename="gen",
        path=main,
    )

    ma.variablesToNtuple(
        decayString="B0:det",
        variables=B0_vars,
        filename=output_filepath,
        treename="det",
        path=main,
    )


append_global_tag()

define_aliases()

input_to_the_path()

reconstruct_generator_level()

reconstruct_detector_level()

treefit()

rest_of_event()

printMCParticles()

B0_vars = create_variable_lists()

save_output(B0_vars)

b2.process(main)

print(b2.statistics)