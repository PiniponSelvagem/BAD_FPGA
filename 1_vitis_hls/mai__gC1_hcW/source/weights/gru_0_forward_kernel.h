#ifndef GRU_0_FORWARD_KERNEL_H
#define GRU_0_FORWARD_KERNEL_H
#include "../types.h"
#include "../size_bgru.h"

// Taken from: q_bidirectional_gru_forward_kernel.h
const gru_weigth_t g_0_forward_kernel[GRU0_KERNEL_SIZE] = {
	0.18827776610851288,
    0.3182152807712555,
    0.06370861083269119,
    -0.42333516478538513,
    -0.22777792811393738,
    0.026958616450428963,
    -0.19237597286701202,
    0.3740920424461365,
    -0.45646291971206665,
    -0.3349764943122864,
    -0.41461431980133057,
    -0.3583112955093384,
    0.16448454558849335,
    -0.5003378987312317,
    0.1150769591331482,
    -0.23147037625312805,
    -0.17525939643383026,
    -0.3232351541519165,
    0.30464744567871094,
    -0.1924726366996765,
    -0.36688321828842163,
    0.21348758041858673,
    -0.0669596940279007,
    -0.10689006745815277,
    0.3499050438404083,
    0.3107980489730835,
    0.2014101892709732,
    -0.38204845786094666,
    -0.23980765044689178,
    -0.21402932703495026,
    -0.19472593069076538,
    0.2593296468257904,
    0.19401001930236816,
    0.04236671328544617,
    -0.5202977657318115,
    0.2860950529575348,
    0.25481465458869934,
    -0.3154352307319641,
    0.055596064776182175,
    -0.2818582355976105,
    -0.27832576632499695,
    0.16428883373737335,
    -0.2625773549079895,
    -0.0975911095738411,
    0.008762911893427372,
    0.21323199570178986,
    0.298317551612854,
    -0.1419530063867569,
    0.28788772225379944,
    -0.09011929482221603,
    0.025949466973543167,
    0.3542565107345581,
    0.2579479217529297,
    0.3986688256263733,
    0.4366491138935089,
    0.07400579005479813,
    -0.015872087329626083,
    -0.4523125886917114,
    0.38757258653640747,
    -0.31925147771835327,
    -0.12338309735059738,
    -0.5419260263442993,
    -0.33754315972328186,
    0.3541276156902313,
    0.3582209646701813,
    0.24256937205791473,
    0.15829205513000488,
    -0.10014400631189346,
    0.22439943253993988,
    -0.008617185987532139,
    0.27321699261665344,
    0.16053567826747894,
    0.11349630355834961,
    0.23965518176555634,
    0.15403366088867188,
    0.21347348392009735,
    0.08052100241184235,
    -0.23496876657009125,
    0.15648505091667175,
    0.3265082538127899,
    0.3274882137775421,
    0.06438830494880676,
    -0.018050558865070343,
    -0.22839386761188507,
    -0.017520714551210403,
    0.34641921520233154,
    0.26619118452072144,
    0.3581547737121582,
    0.34447601437568665,
    0.4439147114753723,
    0.310736745595932,
    0.32288241386413574,
    0.4544214904308319,
    -0.09629780054092407,
    -0.07718093693256378,
    0.19288624823093414,
    0.06740985065698624,
    0.11238256841897964,
    0.26518598198890686,
    -0.2079523354768753,
    0.27918803691864014,
    0.21519087255001068,
    0.5177111625671387,
    0.14871340990066528,
    -0.3031421899795532,
    -0.029269229620695114,
    0.19510848820209503,
    0.0034073179122060537,
    0.13274703919887543,
    0.25793883204460144,
    0.1683829426765442,
    0.0339752659201622,
    0.1139708161354065,
    -0.14089184999465942,
    0.43639156222343445,
    -0.12445715814828873,
    0.1713910698890686,
    0.12782755494117737,
    -0.05986742675304413,
    -0.01929914951324463,
    -0.15828339755535126,
    0.4450395703315735,
    -0.1835523396730423,
    0.3238586187362671,
    -0.029374029487371445,
    -0.04382701590657234,
    0.11825784295797348,
    0.17743107676506042,
    0.21460169553756714,
    -0.34880712628364563,
    -0.40992119908332825,
    0.06632937490940094,
    0.2176588475704193,
    -0.3685835301876068,
    0.18317632377147675,
    0.029772771522402763,
    0.22378775477409363,
    0.2902458906173706,
    0.07306291162967682,
    -0.28327611088752747,
    -0.24322324991226196,
    -0.0884322077035904,
    0.2744738459587097,
    0.021251551806926727,
    -0.22155609726905823,
    0.0938059389591217,
    -0.04750816151499748,
    -0.007325371261686087,
    0.08280269056558609,
    -0.1718427985906601,
    -0.44904157519340515,
    -0.380303293466568,
    -0.33478638529777527,
    0.13888216018676758,
    -0.26519137620925903,
    -0.012990911491215229,
    0.11330485343933105,
    -0.09806882590055466,
    0.06454364955425262,
    -0.37912750244140625,
    -0.19420084357261658,
    0.14611032605171204,
    -0.018134698271751404,
    -0.2362120896577835,
    0.3042432367801666,
    0.2513238489627838,
    0.0980493351817131,
    -0.13051408529281616,
    -0.1595863252878189,
    0.14893831312656403,
    0.0768134742975235,
    0.10064742714166641,
    -0.08677490800619125,
    -0.42180606722831726,
    -0.21993465721607208,
    -0.34460756182670593,
    -0.03589819744229317,
    0.174098402261734,
    0.05205610394477844,
    -0.35610058903694153,
    -0.4390392303466797,
    -0.2931848466396332,
    -0.3472963869571686,
    -0.236463725566864,
    -0.2906428873538971,
    0.1621350198984146,
    -0.3135419189929962,
    0.17604473233222961,
    -0.028734756633639336,
    0.1014653742313385,
    0.27303043007850647,
    -0.18659190833568573,
};

#endif // GRU_0_FORWARD_KERNEL_H