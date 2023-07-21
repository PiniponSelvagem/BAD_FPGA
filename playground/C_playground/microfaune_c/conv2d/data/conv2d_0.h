#pragma once

#ifndef CONV2D_0_H
#define CONV2D_0_H

#include "../../global_settings.h"
#include "../conv2d_settings.h"
#include "../../input/input.h"

#define C2D_0__RAW_IN_LINES     INPUT_LINES
#define C2D_0__RAW_IN_COLS      INPUT_COLS
#define C2D_0__RAW_OUT_LINES    C2D_0__RAW_IN_LINES
#define C2D_0__RAW_OUT_COLS     C2D_0__RAW_IN_COLS

#define C2D_0__IN_LINES         (C2D_0__RAW_IN_LINES + PADDING)
#define C2D_0__IN_COLS          (C2D_0__RAW_IN_COLS + PADDING)

#define C2D_0__OUT_LINES        C2D_0__IN_LINES
#define C2D_0__OUT_COLS         C2D_0__IN_COLS

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const conv_t kernel_0[CHANNELS][C2D_KERNEL_LINES][C2D_KERNEL_COLS] = {
    {
        {
            0.029623493552207947,
            -0.05585840716958046,
            0.005694271996617317
        },
        {
            0.04749192297458649,
            -0.0020606843754649162,
            0.0049833981320261955
        },
        {
            0.025802062824368477,
            -0.06978029757738113,
            0.025392645969986916
        }
    },
    {
        {
            -0.011391389183700085,
            0.014367984607815742,
            0.04414309188723564
        },
        {
            -0.03561719134449959,
            0.07229690253734589,
            -0.03214915096759796
        },
        {
            0.0024907125625759363,
            -0.06375319510698318,
            0.02649715729057789
        }
    },
    {
        {
            0.070817731320858,
            -0.15017984807491302,
            -0.0032803986687213182
        },
        {
            -0.021606380119919777,
            -0.09321277588605881,
            0.07301598787307739
        },
        {
            0.12270388007164001,
            -0.09451402723789215,
            0.1356125771999359
        }
    },
    {
        {
            -0.05838863179087639,
            0.05373632535338402,
            -0.09599016606807709
        },
        {
            0.05170512571930885,
            -0.0012815033551305532,
            0.012490972876548767
        },
        {
            -0.08960655331611633,
            0.04831456393003464,
            0.08030454814434052
        }
    },
    {
        {
            -0.055004481226205826,
            0.00928090326488018,
            -0.06944267451763153
        },
        {
            0.12195838987827301,
            0.0371517576277256,
            -0.10241874307394028
        },
        {
            0.1238151267170906,
            -0.048764750361442566,
            0.013482926413416862
        }
    },
    {
        {
            -0.010610866360366344,
            0.004024932626634836,
            0.007385720498859882
        },
        {
            0.011826520785689354,
            -0.016166651621460915,
            -0.016436582431197166
        },
        {
            -0.01609824225306511,
            -0.012580384500324726,
            0.039222944527864456
        }
    },
    {
        {
            0.10428823530673981,
            0.05065798759460449,
            0.1263982653617859
        },
        {
            -0.07797273248434067,
            -0.07352063804864883,
            -0.02246614173054695
        },
        {
            -0.0763326957821846,
            -0.10373784601688385,
            0.04402133822441101
        }
    },
    {
        {
            0.02529381401836872,
            0.0330810472369194,
            0.055418502539396286
        },
        {
            -0.06855513155460358,
            -0.07749143242835999,
            0.042398273944854736
        },
        {
            0.004405300132930279,
            -0.012986198998987675,
            -0.02829681895673275
        }
    },
    {
        {
            0.057622045278549194,
            -0.060283973813056946,
            0.0012775772484019399
        },
        {
            0.02060110867023468,
            0.00470286188647151,
            -0.0003343771677464247
        },
        {
            -0.07314702123403549,
            0.08439799398183823,
            -0.005632240325212479
        }
    },
    {
        {
            -0.08690106123685837,
            -0.03551241755485535,
            0.09405627846717834
        },
        {
            -0.10288333892822266,
            -0.0695318803191185,
            0.1074887290596962
        },
        {
            0.002861137269064784,
            0.004202092997729778,
            0.05061819776892662
        }
    },
    {
        {
            0.03002193383872509,
            -0.0977337583899498,
            -0.12926732003688812
        },
        {
            0.03473414108157158,
            0.018452240154147148,
            0.07711153477430344
        },
        {
            -0.014957679435610771,
            -0.01619076356291771,
            0.11289597302675247
        }
    },
    {
        {
            0.01398202683776617,
            0.007512410171329975,
            0.051666509360075
        },
        {
            -0.030163900926709175,
            -0.05421632155776024,
            -0.038088105618953705
        },
        {
            -0.026878666132688522,
            0.08905050158500671,
            -0.01988256722688675
        }
    },
    {
        {
            0.07167989015579224,
            -0.018864817917346954,
            0.04653489217162132
        },
        {
            0.04526630416512489,
            -0.030951160937547684,
            -0.03755870833992958
        },
        {
            0.030408307909965515,
            -0.0438581258058548,
            -0.030893787741661072
        }
    },
    {
        {
            -0.027737094089388847,
            -0.01379447616636753,
            0.055717576295137405
        },
        {
            0.009577931836247444,
            -0.04608277231454849,
            0.026086565107107162
        },
        {
            0.05154295638203621,
            -0.05016935244202614,
            0.018083544448018074
        }
    },
    {
        {
            0.038364145904779434,
            0.059437114745378494,
            0.032352857291698456
        },
        {
            -0.0045818849466741085,
            0.016551276668906212,
            -0.0720963254570961
        },
        {
            -0.05758242681622505,
            -0.07460232824087143,
            0.08042297512292862
        }
    },
    {
        {
            0.003089768812060356,
            -0.0392860509455204,
            0.0030137270223349333
        },
        {
            0.014938750304281712,
            0.011177369393408298,
            0.03749321028590202
        },
        {
            0.0018425857415422797,
            -0.028750862926244736,
            0.03966083005070686
        }
    },
    {
        {
            -0.014365680515766144,
            0.020367290824651718,
            -0.05381147190928459
        },
        {
            0.0536319799721241,
            -0.11270845681428909,
            0.007451553363353014
        },
        {
            0.06082465499639511,
            0.02090153656899929,
            -0.021428843960165977
        }
    },
    {
        {
            -0.06196775659918785,
            -0.07421310245990753,
            -0.040786709636449814
        },
        {
            -0.05622376501560211,
            0.030713330954313278,
            0.037280745804309845
        },
        {
            0.10036426037549973,
            0.08471914380788803,
            0.01544824056327343
        }
    },
    {
        {
            0.053052350878715515,
            0.05115121230483055,
            -0.04048986732959747
        },
        {
            0.029823999851942062,
            0.05525679141283035,
            -0.09371736645698547
        },
        {
            -0.05158419907093048,
            0.06126191094517708,
            -0.036166395992040634
        }
    },
    {
        {
            0.14034613966941833,
            0.06807686388492584,
            0.05031350255012512
        },
        {
            -0.004981985781341791,
            -0.018408821895718575,
            -0.0861254557967186
        },
        {
            0.017149146646261215,
            -0.13896189630031586,
            -0.014139940962195396
        }
    },
    {
        {
            0.07559402287006378,
            0.10782024264335632,
            0.04624500870704651
        },
        {
            0.06126973778009415,
            0.04712580516934395,
            -0.09727294743061066
        },
        {
            -0.11156055331230164,
            -0.1321519911289215,
            0.01585089974105358
        }
    },
    {
        {
            0.03738574683666229,
            -0.020288072526454926,
            0.006344761233776808
        },
        {
            -0.004545458126813173,
            0.017449647188186646,
            -0.018803637474775314
        },
        {
            -0.01981167308986187,
            -0.05459296330809593,
            0.08176449686288834
        }
    },
    {
        {
            -0.05098557472229004,
            -0.04026412218809128,
            -0.02799759991466999
        },
        {
            -0.01813141256570816,
            0.024776466190814972,
            0.06158462539315224
        },
        {
            -0.029100937768816948,
            0.018550949171185493,
            0.11646658182144165
        }
    },
    {
        {
            -0.013818254694342613,
            -0.04599601775407791,
            0.06015843152999878
        },
        {
            -0.03890340402722359,
            0.08214487135410309,
            0.06420543044805527
        },
        {
            0.02520236372947693,
            -0.03584437817335129,
            -0.11696592718362808
        }
    },
    {
        {
            0.017677944153547287,
            0.09582752734422684,
            0.07569808512926102
        },
        {
            -0.06273818016052246,
            0.021994750946760178,
            -0.012910708785057068
        },
        {
            -0.0459468849003315,
            0.004711436573415995,
            -0.07322578132152557
        }
    },
    {
        {
            0.0021827879827469587,
            0.0280454121530056,
            0.008075838908553123
        },
        {
            -0.09383363276720047,
            0.05976000055670738,
            -0.055070482194423676
        },
        {
            0.013076722621917725,
            0.04005497694015503,
            1.2334774510236457e-05
        }
    },
    {
        {
            0.07705924659967422,
            0.041179727762937546,
            0.006897689774632454
        },
        {
            0.014959507621824741,
            -0.04095619171857834,
            0.01579274609684944
        },
        {
            -0.030292876064777374,
            -0.04412124305963516,
            -0.06506939232349396
        }
    },
    {
        {
            -0.10738706588745117,
            0.0465666837990284,
            0.08270561695098877
        },
        {
            -0.06509906053543091,
            0.019548960030078888,
            -0.09339374303817749
        },
        {
            0.09728493541479111,
            0.020340388640761375,
            -0.05950229987502098
        }
    },
    {
        {
            0.10632041841745377,
            0.015759676694869995,
            -0.04669763892889023
        },
        {
            0.08595838397741318,
            -0.018240461125969887,
            -0.020594676956534386
        },
        {
            -0.05760379135608673,
            -0.05871516093611717,
            0.005143872462213039
        }
    },
    {
        {
            -0.03234940394759178,
            0.11051838845014572,
            0.07787156850099564
        },
        {
            -0.10477256774902344,
            -0.06350354105234146,
            0.05402926355600357
        },
        {
            -0.08990144729614258,
            -0.027986835688352585,
            0.03361532464623451
        }
    },
    {
        {
            -0.03228934854269028,
            -0.0915931686758995,
            0.047436293214559555
        },
        {
            -0.08157025277614594,
            0.07888279110193253,
            0.0458340123295784
        },
        {
            0.028117729350924492,
            0.0674261748790741,
            -0.018005166202783585
        }
    },
    {
        {
            0.005658613983541727,
            0.0007390370592474937,
            -0.006859310902655125
        },
        {
            0.030002783983945847,
            -0.01185703743249178,
            0.005910168401896954
        },
        {
            -0.028385110199451447,
            0.018590746447443962,
            -0.008896106854081154
        }
    },
    {
        {
            -0.09880514442920685,
            -0.0199089627712965,
            -0.07424765080213547
        },
        {
            0.04552634432911873,
            0.10250069200992584,
            0.012494967319071293
        },
        {
            0.047560472041368484,
            -0.003272920148447156,
            -0.003792827483266592
        }
    },
    {
        {
            0.017565978690981865,
            0.09196770936250687,
            -0.07417469471693039
        },
        {
            0.07516660541296005,
            0.03038647212088108,
            -0.042618364095687866
        },
        {
            -0.0009040645672939718,
            -0.036586739122867584,
            -0.07024556398391724
        }
    },
    {
        {
            -0.08326419442892075,
            0.04697994515299797,
            0.002957842778414488
        },
        {
            -0.07085973769426346,
            0.039265040308237076,
            -0.030428579077124596
        },
        {
            0.04949194937944412,
            0.00597048457711935,
            0.03408181667327881
        }
    },
    {
        {
            -0.0406503863632679,
            -0.14707425236701965,
            0.010645398870110512
        },
        {
            0.020533081144094467,
            0.04091552272439003,
            0.006098730489611626
        },
        {
            0.051037393510341644,
            0.0976019948720932,
            -0.04339572787284851
        }
    },
    {
        {
            0.06975217908620834,
            -0.03363937884569168,
            0.025665296241641045
        },
        {
            -0.08198545128107071,
            -0.005100459326058626,
            0.07834085822105408
        },
        {
            -0.017032308503985405,
            -0.09444934874773026,
            0.10021547228097916
        }
    },
    {
        {
            0.004369198344647884,
            0.018047921359539032,
            -0.07665278762578964
        },
        {
            -0.05594247207045555,
            0.08233872056007385,
            0.07322321087121964
        },
        {
            0.013342867605388165,
            -0.02096070908010006,
            -0.013118007220327854
        }
    },
    {
        {
            -0.03712189197540283,
            -0.005865773186087608,
            0.045885711908340454
        },
        {
            -0.027777306735515594,
            0.016536567360162735,
            -0.09707274287939072
        },
        {
            0.06725914776325226,
            0.03229091316461563,
            -0.0006750888423994184
        }
    },
    {
        {
            0.020084211602807045,
            0.013360322453081608,
            -0.016289453953504562
        },
        {
            0.01928069069981575,
            -0.0412113219499588,
            0.025816578418016434
        },
        {
            0.03235168755054474,
            0.014677520841360092,
            -0.05017074570059776
        }
    },
    {
        {
            0.031346142292022705,
            0.04547747224569321,
            -0.05611255019903183
        },
        {
            -0.08403471857309341,
            -0.07085099816322327,
            0.07826574891805649
        },
        {
            -0.023501329123973846,
            0.032067928463220596,
            0.029186706990003586
        }
    },
    {
        {
            0.0819421261548996,
            0.005387197248637676,
            -0.07147122919559479
        },
        {
            0.08800062537193298,
            0.026192260906100273,
            -0.00016237286035902798
        },
        {
            0.061154503375291824,
            -0.05643328279256821,
            -0.10567374527454376
        }
    },
    {
        {
            -0.1082879826426506,
            -0.05164467915892601,
            0.07110989838838577
        },
        {
            0.036589063704013824,
            0.0495324544608593,
            0.04388059303164482
        },
        {
            0.09320437163114548,
            0.03402787074446678,
            -0.12455809861421585
        }
    },
    {
        {
            -0.010568407364189625,
            -0.10136959701776505,
            0.07768803834915161
        },
        {
            -0.06912418454885483,
            -0.020032167434692383,
            0.010034510865807533
        },
        {
            -0.030900457873940468,
            0.06932590156793594,
            0.08035049587488174
        }
    },
    {
        {
            -0.061681486666202545,
            -0.033664315938949585,
            -0.0026247187051922083
        },
        {
            -0.024120856076478958,
            -0.07741338014602661,
            0.012656131759285927
        },
        {
            0.11202464252710342,
            0.04761794954538345,
            0.05218476057052612
        }
    },
    {
        {
            -0.08818192034959793,
            0.08227486163377762,
            -0.03758787363767624
        },
        {
            -0.05112287774682045,
            0.10305062681436539,
            -0.026524171233177185
        },
        {
            0.06046931445598602,
            -0.060427065938711166,
            0.009872863069176674
        }
    },
    {
        {
            -0.0762924775481224,
            -0.0010012516286224127,
            0.08114014565944672
        },
        {
            0.07579147070646286,
            -0.07524281740188599,
            -0.090870201587677
        },
        {
            0.03629521653056145,
            0.03154342994093895,
            0.034040339291095734
        }
    },
    {
        {
            -0.1429179310798645,
            0.015526576898992062,
            -0.004551599267870188
        },
        {
            9.01215971680358e-05,
            0.02935393527150154,
            -0.01066084485501051
        },
        {
            0.04679426550865173,
            0.10141066461801529,
            -0.03266693651676178
        }
    },
    {
        {
            0.010869074612855911,
            -0.0889507457613945,
            0.07726781815290451
        },
        {
            -0.06714815646409988,
            0.03624836355447769,
            -0.0033499763812869787
        },
        {
            -0.05730738863348961,
            0.033051032572984695,
            0.05512998625636101
        }
    },
    {
        {
            0.13107579946517944,
            -0.033787500113248825,
            -0.018115080893039703
        },
        {
            0.08859562128782272,
            -0.025272579863667488,
            -0.06680100411176682
        },
        {
            -0.015413081273436546,
            -0.052751071751117706,
            -0.08773139864206314
        }
    },
    {
        {
            0.06757999211549759,
            -0.0678986981511116,
            -0.004294572863727808
        },
        {
            0.007822145707905293,
            0.012681449763476849,
            0.02659013122320175
        },
        {
            0.011219817213714123,
            0.013432755134999752,
            -0.04658384621143341
        }
    },
    {
        {
            -0.09940122067928314,
            0.0540987104177475,
            -0.07519901543855667
        },
        {
            -0.02072259411215782,
            0.10135059803724289,
            0.0528409443795681
        },
        {
            0.025514813140034676,
            0.0069698612205684185,
            -0.05109207332134247
        }
    },
    {
        {
            -0.08746147900819778,
            -0.009103250689804554,
            0.024534689262509346
        },
        {
            -0.02285671979188919,
            -0.01621970348060131,
            -0.06825494021177292
        },
        {
            0.08989360928535461,
            0.036231279373168945,
            0.046132080256938934
        }
    },
    {
        {
            -0.04790500923991203,
            -0.04117241129279137,
            -0.06007218733429909
        },
        {
            -0.0382169634103775,
            -0.009083698503673077,
            -0.011926266364753246
        },
        {
            0.0864095464348793,
            0.03988175839185715,
            0.02690121717751026
        }
    },
    {
        {
            -0.0048691765405237675,
            -0.003032110631465912,
            0.020515596494078636
        },
        {
            0.033631499856710434,
            0.024398602545261383,
            -0.01985236257314682
        },
        {
            -0.02848219685256481,
            -0.0054098148830235004,
            -0.010706854984164238
        }
    },
    {
        {
            -0.08020100742578506,
            0.0816885381937027,
            0.01304270327091217
        },
        {
            -0.0763603001832962,
            0.04960823059082031,
            0.06648988276720047
        },
        {
            -0.08112741261720657,
            -0.06371968239545822,
            0.04719914495944977
        }
    },
    {
        {
            -0.14093117415905,
            0.029598597437143326,
            -0.06143338978290558
        },
        {
            0.0871695801615715,
            -0.046906791627407074,
            -0.03141263127326965
        },
        {
            0.08890976011753082,
            0.10062690079212189,
            -0.037173863500356674
        }
    },
    {
        {
            -0.03421367332339287,
            0.052410807460546494,
            0.011604187078773975
        },
        {
            -0.04161475598812103,
            0.028034158051013947,
            -0.006432208698242903
        },
        {
            -0.03720914572477341,
            0.031133610755205154,
            -0.0073332032188773155
        }
    },
    {
        {
            -0.014458982273936272,
            0.03450114279985428,
            0.033284831792116165
        },
        {
            -0.010888656601309776,
            -0.034732792526483536,
            0.0005270271212793887
        },
        {
            -0.00859681237488985,
            0.053260140120983124,
            -0.053986359387636185
        }
    },
    {
        {
            0.10265280306339264,
            -0.03851093351840973,
            0.09188419580459595
        },
        {
            -0.07656414806842804,
            -0.11831297725439072,
            0.08151372522115707
        },
        {
            0.008422059006989002,
            -0.05429614707827568,
            0.027246225625276566
        }
    },
    {
        {
            -0.006532374303787947,
            0.08744481205940247,
            0.04840144142508507
        },
        {
            -0.03633200004696846,
            -0.06157419830560684,
            -0.027191786095499992
        },
        {
            0.0006175945745781064,
            0.03307701274752617,
            -0.07145654410123825
        }
    },
    {
        {
            0.15278267860412598,
            -0.03017975203692913,
            0.0836871787905693
        },
        {
            0.1350107342004776,
            -0.02552560530602932,
            -0.06302137672901154
        },
        {
            -0.08087819814682007,
            -0.034125879406929016,
            -0.10878828167915344
        }
    },
    {
        {
            -0.05590441823005676,
            -0.09620605409145355,
            0.02943866141140461
        },
        {
            0.012441267259418964,
            0.09533743560314178,
            0.01553053967654705
        },
        {
            0.06389932334423065,
            -0.010419212281703949,
            -0.06958233565092087
        }
    },
    {
        {
            0.03079434670507908,
            0.09827099740505219,
            0.0710555836558342
        },
        {
            -0.07500023394823074,
            0.040603674948215485,
            -0.06610884517431259
        },
        {
            -0.08690588176250458,
            0.08155713230371475,
            -0.07282854616641998
        }
    }
};

/**
 * @brief This was taken from "model_json/dump_weights_bias.json". (Channels first)
*/
const conv_t bias_0[C2D_BIAS_SIZE] = {
    0.02698972076177597,
    -0.021239474415779114,
    -0.05574258416891098,
    0.098424531519413,
    0.021682964637875557,
    -0.04326557368040085,
    0.006872390862554312,
    -0.013035526499152184,
    -0.0718320831656456,
    -0.06686368584632874,
    0.06847133487462997,
    -0.060572799295186996,
    0.008555297739803791,
    0.0145342368632555,
    0.09477592259645462,
    -0.004460975062102079,
    -0.07669804245233536,
    -0.006540547125041485,
    -0.10321295261383057,
    0.020464828237891197,
    0.027977101504802704,
    -0.022495107725262642,
    0.039867062121629715,
    -0.037253767251968384,
    -0.02715235948562622,
    0.008455446921288967,
    -0.05692214146256447,
    -0.03826424479484558,
    -0.06922095268964767,
    -0.020355751737952232,
    -0.018391352146863937,
    -0.040662989020347595,
    0.02903147228062153,
    0.002478590002283454,
    0.0015631896676495671,
    -0.0027624766808003187,
    -0.040878597646951675,
    -0.01911655068397522,
    0.024899642914533615,
    0.037580255419015884,
    0.10478170961141586,
    -0.007954688742756844,
    -0.01874353177845478,
    0.05618688836693764,
    0.054864026606082916,
    -0.03433368355035782,
    -0.061565645039081573,
    -0.02441565692424774,
    -0.0020682201720774174,
    -0.01919863373041153,
    -0.013867619447410107,
    -0.03637266159057617,
    0.016896113753318787,
    0.03312743455171585,
    -0.050400007516145706,
    -0.12484799325466156,
    -0.004165058955550194,
    0.04432140663266182,
    0.021348092705011368,
    -0.04725256934762001,
    -0.010505269281566143,
    -0.06631070375442505,
    0.007369477767497301,
    -0.018048156052827835
};

#endif // CONV2D_0_H