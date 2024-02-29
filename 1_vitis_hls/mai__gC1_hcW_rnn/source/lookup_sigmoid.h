#ifndef LOOKUP_SIGMOID_H
#define LOOKUP_SIGMOID_H

#include "types.h"

#define W_SIG 8
#define I_SIG 0
#define SIG_TABLE_SIZE 256

#define SIG_startValue_F    -6.3
#define SIG_endValue_F      5.2
#define SIG_startValue      gru_matrix_t(SIG_startValue_F)
#define SIG_endValue        gru_matrix_t(SIG_endValue_F)

gru_sigmoid_t sigmoidTable[SIG_TABLE_SIZE] = {
    0.0018329389424928053,
    0.001917331061206343,
    0.0020056009520359884,
    0.0020979260612042135,
    0.0021944918856038444,
    0.002295492331529055,
    0.0024011300887740793,
    0.0025116170206990664,
    0.00262717457088005,
    0.0027480341869775037,
    0.0028744377624752445,
    0.0030066380969585188,
    0.003144899375616654,
    0.0032894976686718867,
    0.003440721451451385,
    0.0035988721458342296,
    0.0037642646838188564,
    0.0039372280939690045,
    0.004118106111507527,
    0.0043072578128370825,
    0.004505058275274555,
    0.0047118992627918225,
    0.004928189938558903,
    0.005154357605086046,
    0.005390848472758955,
    0.005638128457555363,
    0.0058966840087212435,
    0.0061670229671708255,
    0.006449675455355138,
    0.00674519479931942,
    0.007054158483638765,
    0.007377169139883965,
    0.007714855569224544,
    0.008067873799722628,
    0.008436908178809245,
    0.00882267250136203,
    0.009225911173720337,
    0.009647400413878324,
    0.010087949487988423,
    0.010548401983184583,
    0.01102963711659677,
    0.011532571080272749,
    0.012058158421549586,
    0.012607393458223884,
    0.013181311727654441,
    0.013780991468693063,
    0.014407555135075413,
    0.015062170938613919,
    0.015746054420215566,
    0.016460470046397137,
    0.01720673282858793,
    0.01798620996209156,
    0.018800322481123472,
    0.019650546925845613,
    0.02053841701678358,
    0.02146552533143074,
    0.02243352497721761,
    0.023444131254350145,
    0.024499123301295583,
    0.025600345714918087,
    0.02674971013643618,
    0.02794919679348874,
    0.029200855987655587,
    0.030506809515781122,
    0.03186925201239531,
    0.03329045219941646,
    0.034772754028154496,
    0.03631857769741519,
    0.03793042053023746,
    0.03961085769047993,
    0.04136254271911715,
    0.043188207868714086,
    0.04509066421312835,
    0.0470728015080545,
    0.04913758777658098,
    0.05128806859249635,
    0.05352736603266858,
    0.055858677268449536,
    0.058285272764747334,
    0.060810494054183886,
    0.0634377510526396,
    0.06617051888151368,
    0.06901233416122553,
    0.071966790739887,
    0.07503753482072659,
    0.078228259451784,
    0.081542698341662,
    0.08498461896577122,
    0.08855781492857702,
    0.09226609754891217,
    0.096113286637505,
    0.1001032004385393,
    0.10423964471037127,
    0.10852640092452014,
    0.11296721356678292,
    0.1175657765298332,
    0.12232571859299997,
    0.12725058799210695,
    0.1323438360903148,
    0.13760880016985627,
    0.1430486853743891,
    0.14866654584238834,
    0.15446526508353467,
    0.16044753566236256,
    0.16661583826643608,
    0.1729724202499286,
    0.17951927375754784,
    0.18625811354814195,
    0.19319035465183937,
    0.2003170900090223,
    0.20763906825356668,
    0.21515667181634612,
    0.22286989553769732,
    0.23077832598909592,
    0.23888112171435597,
    0.24717699460893586,
    0.25566419266206036,
    0.264340484290056,
    0.27320314449020006,
    0.28224894304225995,
    0.2914741349794627,
    0.3008744535416962,
    0.3104451058111549,
    0.32018077121429256,
    0.33007560305385036,
    0.3401232332109256,
    0.3503167801296997,
    0.3606488601667931,
    0.3711116023535904,
    0.38169666658369533,
    0.39239526519945356,
    0.40319818791179307,
    0.4140958299471289,
    0.42507822327448597,
    0.4361350707260127,
    0.4472557827855114,
    0.45842951678320004,
    0.4696452182014345,
    0.48089166376623643,
    0.49215750597383634,
    0.5034313186806076,
    0.514701643369206,
    0.525957035693767,
    0.537186111902897,
    0.5483775947409999,
    0.5595203584361665,
    0.57060347239627,
    0.5816162432537214,
    0.5925482549231629,
    0.6033894063646632,
    0.6141299467771483,
    0.6247605079821361,
    0.6352721337956644,
    0.6456563062257954,
    0.6559049683735384,
    0.6660105439556435,
    0.6759659534078198,
    0.6857646265657878,
    0.6954005119586358,
    0.7048680827836169,
    0.7141623396633922,
    0.7232788103153974,
    0.7322135462882151,
    0.7409631169414035,
    0.7495246008630452,
    0.7578955749333288,
    0.7660741012528396,
    0.7740587121610072,
    0.781848393573562,
    0.7894425668680936,
    0.7968410695441768,
    0.8040441348793117,
    0.8110523707944687,
    0.8178667381336095,
    0.8244885285505654,
    0.8309193421843528,
    0.8371610652907534,
    0.8432158479840378,
    0.8490860822283622,
    0.8547743802038482,
    0.8602835531578896,
    0.8656165908380236,
    0.8707766415888887,
    0.8757669931825486,
    0.8805910544388607,
    0.8852523376807329,
    0.8897544420580757,
    0.8941010377640822,
    0.8982958511581741,
    0.9023426508015463,
    0.9062452344037102,
    0.9100074166717789,
    0.9136330180484034,
    0.9171258543192378,
    0.9204897270665393,
    0.9237284149419306,
    0.9268456657284484,
    0.9298451891596922,
    0.9327306504621293,
    0.9355056645853574,
    0.9381737910843213,
    0.9407385296170611,
    0.9432033160215142,
    0.9455715189351327,
    0.9478464369215821,
    0.9500312960695113,
    0.9521292480292982,
    0.9541433684547387,
    0.9560766558178259,
    0.9579320305660557,
    0.9597123345930383,
    0.9614203309945922,
    0.963058704083936,
    0.964630059641019,
    0.9661369253724866,
    0.9675817515601939,
    0.9689669118775844,
    0.9702947043546304,
    0.9715673524733516,
    0.9727870063772249,
    0.973955744179033,
    0.975075573352886,
    0.9761484321972842,
    0.9771761913571617,
    0.9781606553938774,
    0.9791035643930752,
    0.9800065956012473,
    0.9808713650826838,
    0.9816994293892913,
    0.9824922872365034,
    0.9832513811792055,
    0.9839780992822406,
    0.9846737767806535,
    0.9853396977253985,
    0.98597709661073,
    0.9865871599799803,
    0.9871710280068525,
    0.9877297960497546,
    0.9882645161770677,
    0.9887761986615642,
    0.989265813442503,
    0.9897342915541922,
    0.990182526520068,
    0.9906113757115513,
    0.9910216616711567,
    0.9914141733995006,
    0.9917896676060234,
    0.9921488699233811,
    0.9924924760855949,
    0.992821153070155,
    0.9931355402043835,
    0.9934362502364404,
    0.9937238703714382,
    0.9939989632731956,
    0.9942620680322203,
    0.9945137011005495,
};

#endif  // !LOOKUP_SIGMOID_H