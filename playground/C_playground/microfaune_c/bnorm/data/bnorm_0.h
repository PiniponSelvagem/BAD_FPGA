#pragma once
#include "../../global_settings.h"
#include "../bnorm_settings.h"
#include "../../input/input.h"
#include "../../conv2d/data/conv2d_0.h"

#ifndef BNORM_0_H
#define BNORM_0_H

#define BNORM_0__RAW_IN_LINES     C2D_0__RAW_IN_LINES
#define BNORM_0__RAW_IN_COLS      C2D_0__RAW_IN_COLS
#define BNORM_0__RAW_OUT_LINES    BNORM_0__RAW_IN_LINES
#define BNORM_0__RAW_OUT_COLS     BNORM_0__RAW_IN_COLS

#define BNORM_0__IN_LINES         C2D_0__IN_LINES
#define BNORM_0__IN_COLS          C2D_0__IN_COLS

#define BNORM_0__OUT_LINES        BNORM_0__IN_LINES
#define BNORM_0__OUT_COLS         BNORM_0__IN_COLS


/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const bnorm_t gamma_0[CHANNELS] = {
    0.9056835770606995,
    0.8711891174316406,
    1.2240357398986816,
    0.9145350456237793,
    1.0206674337387085,
    0.9414934515953064,
    1.118019938468933,
    0.9206987619400024,
    0.9041450023651123,
    1.020884394645691,
    1.1022112369537354,
    0.9593868255615234,
    0.9624009132385254,
    0.9392145276069641,
    0.9644696116447449,
    0.9260704517364502,
    0.8999662399291992,
    0.9607303738594055,
    0.942889928817749,
    1.1341819763183594,
    1.1085655689239502,
    0.9288808107376099,
    0.9630264639854431,
    0.99154132604599,
    0.9765371084213257,
    0.8688396215438843,
    0.9149929285049438,
    1.009695053100586,
    0.9611150026321411,
    1.0103923082351685,
    1.0205068588256836,
    0.9115203022956848,
    1.0348968505859375,
    0.9434382319450378,
    0.9061567783355713,
    1.0476148128509521,
    0.9838618040084839,
    0.9421183466911316,
    0.9223282337188721,
    0.8736114501953125,
    0.9596449136734009,
    0.9781603813171387,
    1.0303961038589478,
    0.9630237221717834,
    1.0505094528198242,
    0.9908773899078369,
    1.0257033109664917,
    1.0038961172103882,
    0.9498017430305481,
    0.9845700860023499,
    0.8762207627296448,
    0.9610397815704346,
    0.9462980031967163,
    0.9144641160964966,
    0.9196165800094604,
    1.0009785890579224,
    1.0534306764602661,
    0.9242523908615112,
    0.9304612278938293,
    1.0828883647918701,
    0.9387481808662415,
    1.1714922189712524,
    1.0827534198760986,
    1.0248432159423828
};

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const bnorm_t beta_0[CHANNELS] = {
    -0.043935634195804596,
    -0.03524373844265938,
    0.16144201159477234,
    -0.01953166536986828,
    0.0006480520823970437,
    -0.10327225923538208,
    -0.022949712350964546,
    -0.028215967118740082,
    -0.0358123779296875,
    -0.04288803040981293,
    0.07353956997394562,
    -0.06029019504785538,
    0.07728009670972824,
    0.01453233789652586,
    -0.06003556028008461,
    -0.044103942811489105,
    -0.05492274463176727,
    0.03961613029241562,
    0.005106659606099129,
    0.11467870324850082,
    -0.00550819979980588,
    -0.0522291474044323,
    -0.09136923402547836,
    0.0008545882301405072,
    0.06959497183561325,
    0.010589638724923134,
    0.006628930103033781,
    0.030834585428237915,
    0.12513968348503113,
    -0.10790969431400299,
    0.02101857401430607,
    -0.08901022374629974,
    0.09315413236618042,
    0.030149610713124275,
    0.08555509150028229,
    0.046475544571876526,
    0.01449983473867178,
    -0.020861506462097168,
    0.0008564553572796285,
    -0.10018152743577957,
    -0.012658262625336647,
    0.0916561484336853,
    -0.050305821001529694,
    -0.05504770576953888,
    -0.01241726242005825,
    0.09608843922615051,
    -0.03584924712777138,
    0.15203487873077393,
    -0.014043004252016544,
    -0.002216269029304385,
    -0.06817851215600967,
    0.11979971081018448,
    0.13781440258026123,
    -0.009202143177390099,
    -0.06670960038900375,
    -0.11896549165248871,
    0.030216315761208534,
    -0.09647069126367569,
    -0.07362491637468338,
    0.059798140078783035,
    -0.07373233884572983,
    0.14994792640209198,
    0.04382765665650368,
    0.1471361368894577
};

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const bnorm_t movingmean_0[CHANNELS] = {
    0.02926195226609707,
    -0.016884740442037582,
    -0.04769141972064972,
    0.09915070235729218,
    0.030242430046200752,
    -0.04592767357826233,
    -0.0018921720329672098,
    -0.020648591220378876,
    -0.06396271288394928,
    -0.077659010887146,
    0.07190573960542679,
    -0.06223426014184952,
    0.016537491232156754,
    0.019673191010951996,
    0.09936308115720749,
    0.0063395812176167965,
    -0.08707167953252792,
    0.0029259626753628254,
    -0.09444241225719452,
    0.023690739646553993,
    0.03154771402478218,
    -0.016516366973519325,
    0.05382410064339638,
    -0.0425441637635231,
    -0.021005509421229362,
    0.009633657522499561,
    -0.06345036625862122,
    -0.05326836183667183,
    -0.06629596650600433,
    -0.03220910206437111,
    -0.006793979089707136,
    -0.03925693780183792,
    0.03181520104408264,
    0.0010826661018654704,
    0.0003888260689564049,
    -0.003798028687015176,
    -0.03126384690403938,
    -0.012169357389211655,
    0.023496918380260468,
    0.042280152440071106,
    0.09976322203874588,
    0.00033523840829730034,
    -0.007052054163068533,
    0.056722044944763184,
    0.06082618981599808,
    -0.035776130855083466,
    -0.05787145346403122,
    -0.023004580289125443,
    -0.0036814454942941666,
    -0.04016214236617088,
    -0.008521145209670067,
    -0.036725498735904694,
    0.015022908337414265,
    0.018778417259454727,
    -0.04864872246980667,
    -0.13643208146095276,
    -0.0063374461606144905,
    0.0437995009124279,
    0.021296042948961258,
    -0.04267694056034088,
    -0.018963273614645004,
    -0.058737706393003464,
    0.0034198316279798746,
    -0.01126603689044714
};

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const bnorm_t movingvariance_0[CHANNELS] = {
    3.146245944662951e-05,
    1.2123565284127835e-05,
    0.00012916207197122276,
    3.234539326513186e-05,
    0.0001441670610802248,
    1.878708417279995e-06,
    5.86391834076494e-05,
    1.5457339031854644e-05,
    4.5790620788466185e-06,
    0.00017570366617292166,
    2.5194842237397097e-05,
    7.98358360043494e-06,
    6.75727569614537e-05,
    1.473154316045111e-05,
    1.2653559679165483e-05,
    1.656002496019937e-05,
    5.371718725655228e-05,
    2.9511324100894853e-05,
    3.405569805181585e-05,
    9.765088179847226e-05,
    3.6422636185307056e-05,
    8.589807293901686e-06,
    8.991092181531712e-05,
    9.569943358656019e-06,
    4.358080695965327e-05,
    2.1331234165700153e-05,
    2.9989694667165168e-05,
    3.624456803663634e-05,
    6.590438715647906e-05,
    0.00019409952801652253,
    5.113226507091895e-05,
    5.063292860540969e-07,
    1.5577044905512594e-05,
    7.203083077911288e-05,
    3.578652467695065e-05,
    1.7997946997638792e-05,
    6.124308856669813e-05,
    1.046128454618156e-05,
    6.934185876161791e-06,
    1.7408892745152116e-05,
    2.2360940420185216e-05,
    0.00020000881340820342,
    1.4715465113113169e-05,
    7.729769276920706e-05,
    2.4387816665694118e-05,
    2.14532483369112e-05,
    9.210119969793595e-06,
    3.879043651977554e-05,
    5.99218656134326e-05,
    0.00023674704425502568,
    2.3387956389342435e-05,
    3.4227130527142435e-05,
    1.2622829672181979e-05,
    3.098669185419567e-05,
    1.0507185379537987e-06,
    0.0001891757274279371,
    4.032689321320504e-05,
    3.827458567684516e-05,
    4.723058282252168e-06,
    5.1244012865936384e-05,
    1.7114365618908778e-05,
    0.00016448501264676452,
    9.425795724382624e-06,
    7.160441600717604e-05
};


#endif // BNORM_0_H