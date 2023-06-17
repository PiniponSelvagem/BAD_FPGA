#pragma once
#include "../../global_settings.h"
#include "../bnorm_settings.h"
#include "../../conv2d/data/conv2d_3.h"

#ifndef BNORM_3_H
#define BNORM_3_H

#define BNORM_3__RAW_IN_LINES     C2D_2__RAW_IN_LINES
#define BNORM_3__RAW_IN_COLS      C2D_2__RAW_IN_COLS
#define BNORM_3__RAW_OUT_LINES    BNORM_3__RAW_IN_LINES
#define BNORM_3__RAW_OUT_COLS     BNORM_3__RAW_IN_COLS

#define BNORM_3__IN_LINES         (BNORM_3__RAW_IN_LINES + PADDING)
#define BNORM_3__IN_COLS          (BNORM_3__RAW_IN_COLS + PADDING)

#define BNORM_3__OUT_LINES        BNORM_3__IN_LINES
#define BNORM_3__OUT_COLS         BNORM_3__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const bnorm_t gamma_3[CHANNELS] = {
    0.957618772983551,
    0.9036452174186707,
    0.9588091373443604,
    0.9729727506637573,
    1.061091661453247,
    1.044763445854187,
    0.9866035580635071,
    0.8857448101043701,
    1.0653756856918335,
    0.9079381823539734,
    0.9604360461235046,
    0.9123610258102417,
    0.949503481388092,
    0.8959619998931885,
    0.9574546217918396,
    1.0596880912780762,
    0.954760730266571,
    1.049293041229248,
    1.0272537469863892,
    1.0656553506851196,
    1.0172789096832275,
    1.0319002866744995,
    1.0475634336471558,
    0.8630375266075134,
    0.9626661539077759,
    1.0281490087509155,
    0.9293705821037292,
    0.9758450388908386,
    1.013365387916565,
    0.9647294282913208,
    0.9763548374176025,
    0.8712857365608215,
    1.0315572023391724,
    0.9729741811752319,
    0.9906288981437683,
    1.0212008953094482,
    0.9775579571723938,
    1.0316810607910156,
    1.0124003887176514,
    1.049494743347168,
    0.997392475605011,
    0.8761250376701355,
    1.0402458906173706,
    0.9660810828208923,
    1.0212488174438477,
    0.931428074836731,
    1.0364590883255005,
    0.9647397398948669,
    0.9934852719306946,
    1.0389773845672607,
    0.9139900803565979,
    1.0710126161575317,
    0.97124844789505,
    1.0221443176269531,
    1.0175474882125854,
    1.029137372970581,
    1.0307866334915161,
    0.9265362024307251,
    0.9313017725944519,
    0.9947701096534729,
    0.9778433442115784,
    0.9973377585411072,
    1.0364863872528076,
    0.9257711172103882
};

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const bnorm_t beta_3[CHANNELS] = {
    -0.040694709867239,
    -0.10386771708726883,
    0.006942867301404476,
    0.0017205635085701942,
    -0.021694611757993698,
    0.019179852679371834,
    -0.033360227942466736,
    -0.029131077229976654,
    0.03883938491344452,
    0.034873396158218384,
    -0.038764242082834244,
    -0.04171450808644295,
    -0.02230202965438366,
    -0.09570754319429398,
    -0.013010668568313122,
    0.006066321395337582,
    -0.00045138614950701594,
    -0.014875632710754871,
    -0.10144586861133575,
    0.04931291565299034,
    -0.02791074849665165,
    0.08818817138671875,
    -0.07679178565740585,
    -0.0546419657766819,
    0.029589004814624786,
    -0.014757849276065826,
    -0.05001205578446388,
    -0.04976726695895195,
    -0.017744988203048706,
    -0.008636356331408024,
    -0.03272134065628052,
    0.0013445958029478788,
    -0.032187316566705704,
    -0.052882272750139236,
    -0.019946301355957985,
    -0.016026007011532784,
    -0.065433070063591,
    0.011755106039345264,
    0.08034495264291763,
    -0.005384876858443022,
    -0.009340014308691025,
    -0.03074941411614418,
    -0.013730409555137157,
    0.01703728921711445,
    -0.007839372381567955,
    -0.037420663982629776,
    0.015912676230072975,
    -0.0650755986571312,
    0.010572469793260098,
    -0.042730409651994705,
    -0.09269986301660538,
    -0.015103200450539589,
    -0.09209073334932327,
    -0.02312929555773735,
    -0.03675125539302826,
    0.07270767539739609,
    0.03440915048122406,
    -0.014582730829715729,
    -0.02857893705368042,
    -0.004180060233920813,
    -0.010737582109868526,
    -0.04190947860479355,
    -0.06716877222061157,
    -0.03193880617618561
};

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const bnorm_t movingmean_3[CHANNELS] = {
    -0.07597309350967407,
    0.1263926923274994,
    0.48797857761383057,
    -0.3804987072944641,
    -0.40995168685913086,
    0.06851369142532349,
    0.3515753448009491,
    0.021338127553462982,
    -0.23755007982254028,
    -0.12316104769706726,
    0.12737318873405457,
    -0.19377729296684265,
    0.040277112275362015,
    -0.1476133167743683,
    -0.26816242933273315,
    -0.35669779777526855,
    0.14580608904361725,
    -0.13079853355884552,
    0.46833670139312744,
    0.03165876865386963,
    -0.029370184987783432,
    -0.4792530834674835,
    0.46979936957359314,
    -0.05476513132452965,
    -0.3143470287322998,
    -0.009059147909283638,
    -0.19592061638832092,
    0.028421171009540558,
    -0.3362036943435669,
    -0.34256577491760254,
    -0.22857242822647095,
    0.0785282552242279,
    -0.06882067024707794,
    -0.07335734367370605,
    -0.10723689943552017,
    0.49716731905937195,
    -0.27419641613960266,
    -0.3007686734199524,
    -0.39085930585861206,
    -0.14096106588840485,
    0.03710106760263443,
    -0.03483489900827408,
    0.2944567799568176,
    -0.46512898802757263,
    -0.1345537304878235,
    -0.02864697203040123,
    -0.27674949169158936,
    0.3723055124282837,
    -0.1669929176568985,
    -0.27791133522987366,
    0.15796315670013428,
    -0.19468918442726135,
    -0.03602539002895355,
    -0.3017966151237488,
    0.2912233769893646,
    -0.21350350975990295,
    -0.23185905814170837,
    -0.30751144886016846,
    0.10462601482868195,
    0.15330196917057037,
    -0.30726155638694763,
    0.25603631138801575,
    -0.1560397744178772,
    -0.12713487446308136
};

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const bnorm_t movingvariance_3[CHANNELS] = {
    0.05487554147839546,
    0.09745914489030838,
    0.29600024223327637,
    0.09208817034959793,
    0.31417033076286316,
    0.39027318358421326,
    0.2405186891555786,
    0.031724777072668076,
    0.5845974087715149,
    0.013300050050020218,
    0.14835819602012634,
    0.012372629716992378,
    0.19555142521858215,
    0.1647287756204605,
    0.22445909678936005,
    0.20252491533756256,
    0.17067846655845642,
    0.29891645908355713,
    0.24433448910713196,
    0.5524339079856873,
    0.20119278132915497,
    0.22163929045200348,
    0.31269946694374084,
    0.023924553766846657,
    0.06392107158899307,
    0.22652217745780945,
    0.03286394476890564,
    0.11873321235179901,
    0.47886401414871216,
    0.06379196792840958,
    0.06724792718887329,
    0.06284649670124054,
    0.31468522548675537,
    0.2329777330160141,
    0.15399451553821564,
    0.3339156210422516,
    0.24446572363376617,
    0.3600093722343445,
    0.23083874583244324,
    0.3760669529438019,
    0.2912428379058838,
    0.006459392607212067,
    0.4418219327926636,
    0.0780397430062294,
    0.2737838327884674,
    0.023639116436243057,
    0.48714256286621094,
    0.23439209163188934,
    0.2084932178258896,
    0.34853777289390564,
    0.10473661124706268,
    0.7205445170402527,
    0.08741766959428787,
    0.13628003001213074,
    0.18703323602676392,
    0.3233613967895508,
    0.10825856775045395,
    0.0306768286973238,
    0.16080895066261292,
    0.2751501798629761,
    0.19012834131717682,
    0.2726674973964691,
    0.42111897468566895,
    0.09023553878068924
};


#endif // BNORM_3_H