<project xmlns="com.autoesl.autopilot.project" name="cnn_rnn" top="predict">
    <includePaths/>
    <libraryPaths/>
    <Simulation argv="">
        <SimFlow name="csim" ldflags="" csimMode="0" lastCsimMode="0"/>
    </Simulation>
    <files xmlns="">
        <file name="../cnn_rnn/test_bench/tb_cnn_rnn.cpp" sc="0" tb="1" cflags=" -Wno-unknown-pragmas -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../cnn_rnn/test_bench/data.h" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="cnn_rnn/cnn_rnn/source/cnn_rnn.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_rnn/cnn_rnn/source/settings.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
    </files>
    <solutions xmlns="">
        <solution name="cnn_rnn" status="active"/>
    </solutions>
</project>

