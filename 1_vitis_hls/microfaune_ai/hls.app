<project xmlns="com.autoesl.autopilot.project" name="microfaune_ai" top="predict" projectType="C/C++">
    <includePaths/>
    <libraryPaths/>
    <Simulation>
        <SimFlow name="csim" clean="true" csimMode="0" lastCsimMode="0"/>
    </Simulation>
    <files xmlns="">
        <file name="microfaune_ai/source/axis_bgru.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="microfaune_ai/source/axis_conv3D.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="microfaune_ai/source/global_settings.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="microfaune_ai/source/size_bgru.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="microfaune_ai/source/size_conv3D.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="microfaune_ai/source/types.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="../source/load_weights.h" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="../source/tb_main.cpp" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
    </files>
    <solutions xmlns="">
        <solution name="algorithm" status="active"/>
    </solutions>
</project>

