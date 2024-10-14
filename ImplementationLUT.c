#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <float.h>
#include <immintrin.h>
#include "header.h"


// LOOKUP TABLES: 


/**
 * @brief Lookup table with even spacing
 */
const float lut_even [513] = {
-0x1.200000p+3 , -0x1.000000p+3 , -0x1.da8ffap+2 , -0x1.c00000p+2 , -0x1.ab6588p+2 , -0x1.9a8ffap+2 , -0x1.8c544cp+2 , -0x1.800000p+2 ,
-0x1.751ff2p+2 , -0x1.6b6588p+2 , -0x1.6298acp+2 , -0x1.5a8ffap+2 , -0x1.532bfep+2 , -0x1.4c544cp+2 , -0x1.45f582p+2 , -0x1.400000p+2 ,
-0x1.3a6702p+2 , -0x1.351ff2p+2 , -0x1.3021f4p+2 , -0x1.2b6588p+2 , -0x1.26e446p+2 , -0x1.2298acp+2 , -0x1.1e7df6p+2 , -0x1.1a8ffap+2 ,
-0x1.16cb10p+2 , -0x1.132bfep+2 , -0x1.0fafecp+2 , -0x1.0c544cp+2 , -0x1.0916d6p+2 , -0x1.05f582p+2 , -0x1.02ee72p+2 , -0x1.000000p+2 ,
-0x1.fa514cp+1 , -0x1.f4ce04p+1 , -0x1.ef73a8p+1 , -0x1.ea3fe6p+1 , -0x1.e530a2p+1 , -0x1.e043eap+1 , -0x1.db77f0p+1 , -0x1.d6cb10p+1 ,
-0x1.d23bbcp+1 , -0x1.cdc88ap+1 , -0x1.c97028p+1 , -0x1.c53158p+1 , -0x1.c10af6p+1 , -0x1.bcfbecp+1 , -0x1.b9033cp+1 , -0x1.b51ff2p+1 ,
-0x1.b15130p+1 , -0x1.ad961ep+1 , -0x1.a9edf8p+1 , -0x1.a657fep+1 , -0x1.a2d380p+1 , -0x1.9f5fd8p+1 , -0x1.9bfc68p+1 , -0x1.98a898p+1 ,
-0x1.9563dcp+1 , -0x1.922daep+1 , -0x1.8f058ep+1 , -0x1.8beb02p+1 , -0x1.88dd9ap+1 , -0x1.85dce6p+1 , -0x1.82e87ep+1 , -0x1.800000p+1 ,
-0x1.7d230ep+1 , -0x1.7a514cp+1 , -0x1.778a64p+1 , -0x1.74ce04p+1 , -0x1.721bdep+1 , -0x1.6f73a8p+1 , -0x1.6cd516p+1 , -0x1.6a3fe6p+1 ,
-0x1.67b3d4p+1 , -0x1.6530a2p+1 , -0x1.62b612p+1 , -0x1.6043eap+1 , -0x1.5dd9f0p+1 , -0x1.5b77f0p+1 , -0x1.591db6p+1 , -0x1.56cb10p+1 ,
-0x1.547fccp+1 , -0x1.523bbcp+1 , -0x1.4ffeb6p+1 , -0x1.4dc88ap+1 , -0x1.4b9914p+1 , -0x1.497028p+1 , -0x1.474da0p+1 , -0x1.453158p+1 ,
-0x1.431b2ap+1 , -0x1.410af6p+1 , -0x1.3f0096p+1 , -0x1.3cfbecp+1 , -0x1.3afcd8p+1 , -0x1.39033cp+1 , -0x1.370ef8p+1 , -0x1.351ff2p+1 ,
-0x1.33360ep+1 , -0x1.315130p+1 , -0x1.2f713ep+1 , -0x1.2d961ep+1 , -0x1.2bbfbap+1 , -0x1.29edf8p+1 , -0x1.2820c0p+1 , -0x1.2657fep+1 ,
-0x1.24939ap+1 , -0x1.22d380p+1 , -0x1.21179cp+1 , -0x1.1f5fd8p+1 , -0x1.1dac22p+1 , -0x1.1bfc68p+1 , -0x1.1a5094p+1 , -0x1.18a898p+1 ,
-0x1.170460p+1 , -0x1.1563dcp+1 , -0x1.13c6fcp+1 , -0x1.122daep+1 , -0x1.1097e4p+1 , -0x1.0f058ep+1 , -0x1.0d769cp+1 , -0x1.0beb02p+1 ,
-0x1.0a62b0p+1 , -0x1.08dd9ap+1 , -0x1.075bb0p+1 , -0x1.05dce6p+1 , -0x1.04612ep+1 , -0x1.02e87ep+1 , -0x1.0172c8p+1 , -0x1.000000p+1 ,
-0x1.fd2036p+0 , -0x1.fa461ap+0 , -0x1.f77198p+0 , -0x1.f4a296p+0 , -0x1.f1d902p+0 , -0x1.ef14c8p+0 , -0x1.ec55d0p+0 , -0x1.e99c0ap+0 ,
-0x1.e6e75ep+0 , -0x1.e437bep+0 , -0x1.e18d14p+0 , -0x1.dee74ep+0 , -0x1.dc465cp+0 , -0x1.d9aa2cp+0 , -0x1.d712acp+0 , -0x1.d47fccp+0 ,
-0x1.d1f17ap+0 , -0x1.cf67a8p+0 , -0x1.cce246p+0 , -0x1.ca6144p+0 , -0x1.c7e492p+0 , -0x1.c56c24p+0 , -0x1.c2f7e8p+0 , -0x1.c087d2p+0 ,
-0x1.be1bd4p+0 , -0x1.bbb3e0p+0 , -0x1.b94feap+0 , -0x1.b6efe2p+0 , -0x1.b493bcp+0 , -0x1.b23b6cp+0 , -0x1.afe6e8p+0 , -0x1.ad961ep+0 ,
-0x1.ab4908p+0 , -0x1.a8ff98p+0 , -0x1.a6b9c0p+0 , -0x1.a47778p+0 , -0x1.a238b6p+0 , -0x1.9ffd6ap+0 , -0x1.9dc58ep+0 , -0x1.9b9116p+0 ,
-0x1.995ff8p+0 , -0x1.973228p+0 , -0x1.95079ep+0 , -0x1.92e050p+0 , -0x1.90bc34p+0 , -0x1.8e9b42p+0 , -0x1.8c7d6ep+0 , -0x1.8a62b0p+0 ,
-0x1.884b00p+0 , -0x1.863656p+0 , -0x1.8424a6p+0 , -0x1.8215eap+0 , -0x1.800a1ap+0 , -0x1.7e012cp+0 , -0x1.7bfb18p+0 , -0x1.79f7d8p+0 ,
-0x1.77f762p+0 , -0x1.75f9b0p+0 , -0x1.73febap+0 , -0x1.720678p+0 , -0x1.7010e2p+0 , -0x1.6e1df2p+0 , -0x1.6c2da0p+0 , -0x1.6a3fe6p+0 ,
-0x1.6854bcp+0 , -0x1.666c1cp+0 , -0x1.648600p+0 , -0x1.62a260p+0 , -0x1.60c136p+0 , -0x1.5ee27cp+0 , -0x1.5d062cp+0 , -0x1.5b2c3ep+0 ,
-0x1.5954aep+0 , -0x1.577f74p+0 , -0x1.55ac8cp+0 , -0x1.53dbeep+0 , -0x1.520d98p+0 , -0x1.504180p+0 , -0x1.4e77a4p+0 , -0x1.4caffcp+0 ,
-0x1.4aea84p+0 , -0x1.492734p+0 , -0x1.47660cp+0 , -0x1.45a702p+0 , -0x1.43ea12p+0 , -0x1.422f38p+0 , -0x1.40766ep+0 , -0x1.3ebfb2p+0 ,
-0x1.3d0afap+0 , -0x1.3b5846p+0 , -0x1.39a78ep+0 , -0x1.37f8d0p+0 , -0x1.364c04p+0 , -0x1.34a12ap+0 , -0x1.32f83ap+0 , -0x1.315130p+0 ,
-0x1.2fac0ap+0 , -0x1.2e08c0p+0 , -0x1.2c6752p+0 , -0x1.2ac7b8p+0 , -0x1.2929f0p+0 , -0x1.278df6p+0 , -0x1.25f3c6p+0 , -0x1.245b5cp+0 ,
-0x1.22c4b2p+0 , -0x1.212fc8p+0 , -0x1.1f9c96p+0 , -0x1.1e0b1ap+0 , -0x1.1c7b52p+0 , -0x1.1aed3ap+0 , -0x1.1960cap+0 , -0x1.17d604p+0 ,
-0x1.164ce2p+0 , -0x1.14c560p+0 , -0x1.133f7cp+0 , -0x1.11bb32p+0 , -0x1.10387ep+0 , -0x1.0eb75ep+0 , -0x1.0d37cep+0 , -0x1.0bb9cap+0 ,
-0x1.0a3d50p+0 , -0x1.08c25cp+0 , -0x1.0748ecp+0 , -0x1.05d0fcp+0 , -0x1.045a88p+0 , -0x1.02e590p+0 , -0x1.01720ep+0 , -0x1.000000p+0 ,
-0x1.fd1ec8p-1 , -0x1.fa406cp-1 , -0x1.f764e8p-1 , -0x1.f48c34p-1 , -0x1.f1b64ep-1 , -0x1.eee32ep-1 , -0x1.ec12d0p-1 , -0x1.e9452cp-1 ,
-0x1.e67a40p-1 , -0x1.e3b206p-1 , -0x1.e0ec76p-1 , -0x1.de298ep-1 , -0x1.db694ap-1 , -0x1.d8aba0p-1 , -0x1.d5f090p-1 , -0x1.d33812p-1 ,
-0x1.d08222p-1 , -0x1.cdcebep-1 , -0x1.cb1ddcp-1 , -0x1.c86f7cp-1 , -0x1.c5c396p-1 , -0x1.c31a28p-1 , -0x1.c0732cp-1 , -0x1.bdce9ep-1 ,
-0x1.bb2c7ap-1 , -0x1.b88cbap-1 , -0x1.b5ef5ap-1 , -0x1.b35458p-1 , -0x1.b0bbaep-1 , -0x1.ae2558p-1 , -0x1.ab9152p-1 , -0x1.a8ff98p-1 ,
-0x1.a67024p-1 , -0x1.a3e2f4p-1 , -0x1.a15804p-1 , -0x1.9ecf50p-1 , -0x1.9c48d4p-1 , -0x1.99c48cp-1 , -0x1.974274p-1 , -0x1.94c288p-1 ,
-0x1.9244c4p-1 , -0x1.8fc924p-1 , -0x1.8d4fa8p-1 , -0x1.8ad846p-1 , -0x1.886300p-1 , -0x1.85efd0p-1 , -0x1.837eb4p-1 , -0x1.810fa6p-1 ,
-0x1.7ea2a2p-1 , -0x1.7c37aap-1 , -0x1.79ceb4p-1 , -0x1.7767c2p-1 , -0x1.7502ccp-1 , -0x1.729fd2p-1 , -0x1.703ed0p-1 , -0x1.6ddfc2p-1 ,
-0x1.6b82a6p-1 , -0x1.692778p-1 , -0x1.66ce34p-1 , -0x1.6476dap-1 , -0x1.622162p-1 , -0x1.5fcdcep-1 , -0x1.5d7c18p-1 , -0x1.5b2c3ep-1 ,
-0x1.58de3cp-1 , -0x1.569210p-1 , -0x1.5447b8p-1 , -0x1.51ff2ep-1 , -0x1.4fb872p-1 , -0x1.4d7380p-1 , -0x1.4b3056p-1 , -0x1.48eef2p-1 ,
-0x1.46af4ep-1 , -0x1.44716ap-1 , -0x1.423542p-1 , -0x1.3ffad4p-1 , -0x1.3dc21ep-1 , -0x1.3b8b1cp-1 , -0x1.3955ccp-1 , -0x1.37222cp-1 ,
-0x1.34f038p-1 , -0x1.32bfeep-1 , -0x1.30914cp-1 , -0x1.2e6450p-1 , -0x1.2c38f6p-1 , -0x1.2a0f3cp-1 , -0x1.27e720p-1 , -0x1.25c0a0p-1 ,
-0x1.239bbap-1 , -0x1.217868p-1 , -0x1.1f56acp-1 , -0x1.1d3682p-1 , -0x1.1b17e8p-1 , -0x1.18fadcp-1 , -0x1.16df5ap-1 , -0x1.14c560p-1 ,
-0x1.12aceep-1 , -0x1.109602p-1 , -0x1.0e8096p-1 , -0x1.0c6caap-1 , -0x1.0a5a3ep-1 , -0x1.08494cp-1 , -0x1.0639d4p-1 , -0x1.042bd4p-1 ,
-0x1.021f4ap-1 , -0x1.001434p-1 , -0x1.fc151cp-2 , -0x1.f804aep-2 , -0x1.f3f71cp-2 , -0x1.efec62p-2 , -0x1.ebe47ap-2 , -0x1.e7df60p-2 ,
-0x1.e3dd12p-2 , -0x1.dfdd8ap-2 , -0x1.dbe0c6p-2 , -0x1.d7e6c0p-2 , -0x1.d3ef78p-2 , -0x1.cffae6p-2 , -0x1.cc0908p-2 , -0x1.c819dcp-2 ,
-0x1.c42d5cp-2 , -0x1.c04386p-2 , -0x1.bc5c54p-2 , -0x1.b877c6p-2 , -0x1.b495d4p-2 , -0x1.b0b680p-2 , -0x1.acd9c2p-2 , -0x1.a8ff98p-2 ,
-0x1.a527fep-2 , -0x1.a152f2p-2 , -0x1.9d806ep-2 , -0x1.99b072p-2 , -0x1.95e2fap-2 , -0x1.921800p-2 , -0x1.8e4f84p-2 , -0x1.8a8980p-2 ,
-0x1.86c5f4p-2 , -0x1.8304dap-2 , -0x1.7f462ep-2 , -0x1.7b89f0p-2 , -0x1.77d01cp-2 , -0x1.7418acp-2 , -0x1.7063a2p-2 , -0x1.6cb0f6p-2 ,
-0x1.6900a8p-2 , -0x1.6552b4p-2 , -0x1.61a718p-2 , -0x1.5dfdd0p-2 , -0x1.5a56d8p-2 , -0x1.56b22ep-2 , -0x1.530fd0p-2 , -0x1.4f6fbcp-2 ,
-0x1.4bd1ecp-2 , -0x1.48365ep-2 , -0x1.449d12p-2 , -0x1.410602p-2 , -0x1.3d712cp-2 , -0x1.39de8ep-2 , -0x1.364e26p-2 , -0x1.32bfeep-2 ,
-0x1.2f33e6p-2 , -0x1.2baa0cp-2 , -0x1.28225cp-2 , -0x1.249cd2p-2 , -0x1.21196ep-2 , -0x1.1d982cp-2 , -0x1.1a190ap-2 , -0x1.169c06p-2 ,
-0x1.13211ap-2 , -0x1.0fa848p-2 , -0x1.0c318ap-2 , -0x1.08bce0p-2 , -0x1.054a48p-2 , -0x1.01d9bcp-2 , -0x1.fcd678p-3 , -0x1.f5fd8ap-3 ,
-0x1.ef28aap-3 , -0x1.e857d4p-3 , -0x1.e18b00p-3 , -0x1.dac22ep-3 , -0x1.d3fd54p-3 , -0x1.cd3c72p-3 , -0x1.c67f80p-3 , -0x1.bfc67ap-3 ,
-0x1.b9115ep-3 , -0x1.b26024p-3 , -0x1.abb2cap-3 , -0x1.a5094cp-3 , -0x1.9e63a2p-3 , -0x1.97c1ccp-3 , -0x1.9123c2p-3 , -0x1.8a8980p-3 ,
-0x1.83f304p-3 , -0x1.7d604ap-3 , -0x1.76d14ap-3 , -0x1.704604p-3 , -0x1.69be70p-3 , -0x1.633a8cp-3 , -0x1.5cba54p-3 , -0x1.563dc2p-3 ,
-0x1.4fc4d4p-3 , -0x1.494f86p-3 , -0x1.42ddd2p-3 , -0x1.3c6fb6p-3 , -0x1.36052ep-3 , -0x1.2f9e32p-3 , -0x1.293ac4p-3 , -0x1.22dadcp-3 ,
-0x1.1c7e78p-3 , -0x1.162594p-3 , -0x1.0fd02ap-3 , -0x1.097e38p-3 , -0x1.032fbcp-3 , -0x1.f9c95ep-4 , -0x1.ed3a1ep-4 , -0x1.e0b1aep-4 ,
-0x1.d4300ap-4 , -0x1.c7b528p-4 , -0x1.bb4102p-4 , -0x1.aed392p-4 , -0x1.a26ccep-4 , -0x1.960cb0p-4 , -0x1.89b330p-4 , -0x1.7d604ap-4 ,
-0x1.7113f4p-4 , -0x1.64ce26p-4 , -0x1.588edep-4 , -0x1.4c5610p-4 , -0x1.4023b8p-4 , -0x1.33f7cep-4 , -0x1.27d24cp-4 , -0x1.1bb32ap-4 ,
-0x1.0f9a64p-4 , -0x1.0387f0p-4 , -0x1.eef792p-5 , -0x1.d6ebd2p-5 , -0x1.beec92p-5 , -0x1.a6f9c4p-5 , -0x1.8f135cp-5 , -0x1.77394cp-5 ,
-0x1.5f6b8ap-5 , -0x1.47aa08p-5 , -0x1.2ff4b8p-5 , -0x1.184b8ep-5 , -0x1.00ae80p-5 , -0x1.d23afcp-6 , -0x1.a330fep-6 , -0x1.743ee8p-6 ,
-0x1.4564a6p-6 , -0x1.16a21ep-6 , -0x1.cfee70p-7 , -0x1.72c7bap-7 , -0x1.15cfe8p-7 , -0x1.720d9cp-8 , -0x1.71b0eap-9 , 0x0.000000p+0 ,
0}; // 1 extra lut value for the 3 points polynomial interpolation (not included in lookup)


/**
 * @brief Pre-lookup breakpoints x with uneven spacing
 */
const float bps_uneven [513] = {
0x1.00000000000000p-16 , 0x1.0fffffffffff10p-16 , 0x1.20000000000000p-16 , 0x1.300000000000f0p-16 , 0x1.40000000000000p-16 , 0x1.4fffffffffff10p-16 , 0x1.60000000000000p-16 , 0x1.700000000000f0p-16 ,
0x1.80000000000000p-16 , 0x1.8fffffffffff10p-16 , 0x1.a0000000000000p-16 , 0x1.b00000000000f0p-16 , 0x1.c0000000000000p-16 , 0x1.cfffffffffff10p-16 , 0x1.e0000000000000p-16 , 0x1.f00000000000f0p-16 ,
0x1.00000000000000p-15 , 0x1.07ffffffffff90p-15 , 0x1.10000000000000p-15 , 0x1.18000000000070p-15 , 0x1.20000000000000p-15 , 0x1.27ffffffffff90p-15 , 0x1.30000000000000p-15 , 0x1.38000000000070p-15 ,
0x1.40000000000000p-15 , 0x1.47ffffffffff90p-15 , 0x1.50000000000000p-15 , 0x1.58000000000070p-15 , 0x1.60000000000000p-15 , 0x1.67ffffffffff90p-15 , 0x1.70000000000000p-15 , 0x1.78000000000070p-15 ,
0x1.80000000000000p-15 , 0x1.87ffffffffff90p-15 , 0x1.90000000000000p-15 , 0x1.98000000000070p-15 , 0x1.a0000000000000p-15 , 0x1.a7ffffffffff90p-15 , 0x1.b0000000000000p-15 , 0x1.b8000000000070p-15 ,
0x1.c0000000000000p-15 , 0x1.c7ffffffffff90p-15 , 0x1.d8000000000070p-15 , 0x1.e7ffffffffff90p-15 , 0x1.f8000000000070p-15 , 0x1.03ffffffffffc0p-14 , 0x1.0c000000000040p-14 , 0x1.13ffffffffffc0p-14 ,
0x1.1c000000000040p-14 , 0x1.23ffffffffffc0p-14 , 0x1.2c000000000040p-14 , 0x1.33ffffffffffc0p-14 , 0x1.3c000000000040p-14 , 0x1.43ffffffffffc0p-14 , 0x1.50000000000000p-14 , 0x1.5c000000000040p-14 ,
0x1.68000000000000p-14 , 0x1.73ffffffffffc0p-14 , 0x1.80000000000000p-14 , 0x1.8c000000000040p-14 , 0x1.98000000000000p-14 , 0x1.a3fffffffffee0p-14 , 0x1.b0000000000000p-14 , 0x1.bc000000000120p-14 ,
0x1.c7fffffffffdb0p-14 , 0x1.d3fffffffffee0p-14 , 0x1.e3fffffffffee0p-14 , 0x1.f3fffffffffee0p-14 , 0x1.01ffffffffff70p-13 , 0x1.09ffffffffff70p-13 , 0x1.11ffffffffff70p-13 , 0x1.19ffffffffff70p-13 ,
0x1.21ffffffffff70p-13 , 0x1.2c000000000120p-13 , 0x1.36000000000090p-13 , 0x1.40000000000000p-13 , 0x1.49ffffffffff70p-13 , 0x1.53fffffffffee0p-13 , 0x1.5e000000000090p-13 , 0x1.68000000000000p-13 ,
0x1.73fffffffffee0p-13 , 0x1.80000000000000p-13 , 0x1.8c000000000120p-13 , 0x1.98000000000000p-13 , 0x1.a3fffffffffee0p-13 , 0x1.b1ffffffffff70p-13 , 0x1.c0000000000000p-13 , 0x1.ce000000000090p-13 ,
0x1.dc000000000120p-13 , 0x1.e9ffffffffff70p-13 , 0x1.f9ffffffffff70p-13 , 0x1.04ffffffffffb0p-12 , 0x1.0cffffffffffb0p-12 , 0x1.14ffffffffffb0p-12 , 0x1.1e000000000090p-12 , 0x1.27000000000050p-12 ,
0x1.30000000000000p-12 , 0x1.38ffffffffffb0p-12 , 0x1.43000000000050p-12 , 0x1.4cffffffffffb0p-12 , 0x1.57000000000050p-12 , 0x1.61ffffffffff70p-12 , 0x1.6cffffffffffb0p-12 , 0x1.78000000000000p-12 ,
0x1.84000000000000p-12 , 0x1.90000000000000p-12 , 0x1.9c000000000000p-12 , 0x1.a8ffffffffffb0p-12 , 0x1.b6000000000090p-12 , 0x1.c3000000000050p-12 , 0x1.d0ffffffffffb0p-12 , 0x1.df000000000050p-12 ,
0x1.ee000000000090p-12 , 0x1.fcffffffffffb0p-12 , 0x1.067fffffffffe0p-11 , 0x1.0e7fffffffffe0p-11 , 0x1.167fffffffffe0p-11 , 0x1.1f000000000050p-11 , 0x1.27800000000020p-11 , 0x1.307fffffffffe0p-11 ,
0x1.39800000000020p-11 , 0x1.43000000000050p-11 , 0x1.4cffffffffffb0p-11 , 0x1.57000000000050p-11 , 0x1.61800000000020p-11 , 0x1.6c000000000000p-11 , 0x1.77000000000050p-11 , 0x1.827fffffffffe0p-11 ,
0x1.8e000000000000p-11 , 0x1.9a000000000000p-11 , 0x1.a67fffffffffe0p-11 , 0x1.b3000000000050p-11 , 0x1.c0000000000000p-11 , 0x1.cd800000000020p-11 , 0x1.db000000000050p-11 , 0x1.e8ffffffffffb0p-11 ,
0x1.f7800000000020p-11 , 0x1.033ffffffffff0p-10 , 0x1.0b000000000170p-10 , 0x1.12c00000000060p-10 , 0x1.1ac00000000060p-10 , 0x1.23000000000170p-10 , 0x1.2b8000000000c0p-10 , 0x1.34400000000110p-10 ,
0x1.3d3fffffffffa0p-10 , 0x1.467fffffffff40p-10 , 0x1.50000000000000p-10 , 0x1.59bffffffffef0p-10 , 0x1.63bffffffffef0p-10 , 0x1.6e000000000000p-10 , 0x1.787fffffffff40p-10 , 0x1.833fffffffffa0p-10 ,
0x1.8e7fffffffff40p-10 , 0x1.9a000000000000p-10 , 0x1.a5bffffffffef0p-10 , 0x1.b1bffffffffef0p-10 , 0x1.be400000000110p-10 , 0x1.cb000000000170p-10 , 0x1.d8000000000000p-10 , 0x1.e58000000000c0p-10 ,
0x1.f33fffffffffa0p-10 , 0x1.00c00000000060p-9  , 0x1.08000000000000p-9  , 0x1.0f8000000000c0p-9  , 0x1.17200000000090p-9  , 0x1.1f000000000000p-9  , 0x1.27000000000000p-9  , 0x1.2f3fffffffffa0p-9  ,
0x1.37c00000000060p-9  , 0x1.407fffffffff40p-9  , 0x1.49600000000030p-9  , 0x1.527fffffffff40p-9  , 0x1.5bdfffffffff70p-9  , 0x1.658000000000c0p-9  , 0x1.6f600000000030p-9  , 0x1.798000000000c0p-9  ,
0x1.83dfffffffff70p-9  , 0x1.8e7fffffffff40p-9  , 0x1.99600000000030p-9  , 0x1.a47fffffffff40p-9  , 0x1.b0000000000000p-9  , 0x1.bbc00000000060p-9  , 0x1.c7c00000000060p-9  , 0x1.d4200000000090p-9  ,
0x1.e0c00000000060p-9  , 0x1.edc00000000060p-9  , 0x1.fb000000000000p-9  , 0x1.044ffffffffff0p-8  , 0x1.0b4ffffffffff0p-8  , 0x1.126fffffffffc0p-8  , 0x1.19c00000000060p-8  , 0x1.213fffffffffa0p-8  ,
0x1.28efffffffffc0p-8  , 0x1.30cffffffffff0p-8  , 0x1.38e00000000030p-8  , 0x1.411fffffffffd0p-8  , 0x1.49900000000040p-8  , 0x1.52300000000010p-8  , 0x1.5b100000000040p-8  , 0x1.641fffffffffd0p-8  ,
0x1.6d600000000030p-8  , 0x1.76e00000000030p-8  , 0x1.809fffffffffd0p-8  , 0x1.8a900000000040p-8  , 0x1.94c00000000060p-8  , 0x1.9f300000000010p-8  , 0x1.a9e00000000030p-8  , 0x1.b4cffffffffff0p-8  ,
0x1.c0000000000000p-8  , 0x1.cb6fffffffffc0p-8  , 0x1.d71fffffffffd0p-8  , 0x1.e3100000000040p-8  , 0x1.ef4ffffffffff0p-8  , 0x1.fbcffffffffff0p-8  , 0x1.044ffffffffff0p-7  , 0x1.0ad80000000010p-7  ,
0x1.11880000000020p-7  , 0x1.18600000000030p-7  , 0x1.1f600000000030p-7  , 0x1.26880000000020p-7  , 0x1.2dd80000000010p-7  , 0x1.354ffffffffff0p-7  , 0x1.3cf7ffffffffe0p-7  , 0x1.44c80000000020p-7  ,
0x1.4cc80000000190p-7  , 0x1.54f00000000070p-7  , 0x1.5d480000000190p-7  , 0x1.65d00000000160p-7  , 0x1.6e87ffffffffc0p-7  , 0x1.77700000000070p-7  , 0x1.8087ffffffffc0p-7  , 0x1.89d00000000160p-7  ,
0x1.93500000000160p-7  , 0x1.9d000000000000p-7  , 0x1.a6e800000000b0p-7  , 0x1.b107ffffffffc0p-7  , 0x1.bb6000000000e0p-7  , 0x1.c5f00000000070p-7  , 0x1.d0b7fffffffe70p-7  , 0x1.dbb7fffffffe70p-7  ,
0x1.e6f80000000040p-7  , 0x1.f2700000000070p-7  , 0x1.fe27fffffffee0p-7  , 0x1.050fffffffff90p-6  , 0x1.0b2800000000b0p-6  , 0x1.116400000000d0p-6  , 0x1.17c00000000000p-6  , 0x1.1e3c0000000020p-6  ,
0x1.24dbffffffff30p-6  , 0x1.2b9bffffffff30p-6  , 0x1.32800000000000p-6  , 0x1.3987ffffffffc0p-6  , 0x1.40b40000000050p-6  , 0x1.4803ffffffffe0p-6  , 0x1.4f7c0000000020p-6  , 0x1.5717ffffffff50p-6  ,
0x1.5edbffffffff30p-6  , 0x1.66c7ffffffffc0p-6  , 0x1.6edbffffffff30p-6  , 0x1.7717ffffffff50p-6  , 0x1.7f7c0000000020p-6  , 0x1.880bffffffffb0p-6  , 0x1.90c3ffffffffe0p-6  , 0x1.99a800000000b0p-6  ,
0x1.a2b80000000040p-6  , 0x1.abf40000000050p-6  , 0x1.b56000000000e0p-6  , 0x1.bef80000000040p-6  , 0x1.c8c00000000000p-6  , 0x1.d2b80000000040p-6  , 0x1.dce000000000e0p-6  , 0x1.e7380000000040p-6  ,
0x1.f1c3ffffffffe0p-6  , 0x1.fc800000000000p-6  , 0x1.03b80000000040p-5  , 0x1.0949ffffffffb0p-5  , 0x1.0ef60000000050p-5  , 0x1.14be0000000010p-5  , 0x1.1aa00000000000p-5  , 0x1.209e0000000010p-5  ,
0x1.26b80000000040p-5  , 0x1.2cedffffffffa0p-5  , 0x1.3341fffffffff0p-5  , 0x1.39b20000000060p-5  , 0x1.40400000000000p-5  , 0x1.46ebffffffffb0p-5  , 0x1.4db60000000050p-5  , 0x1.54a00000000000p-5  ,
0x1.5ba7ffffffffc0p-5  , 0x1.62cfffffffff90p-5  , 0x1.6a180000000040p-5  , 0x1.71800000000000p-5  , 0x1.7909ffffffffb0p-5  , 0x1.80b40000000050p-5  , 0x1.88800000000000p-5  , 0x1.906dffffffffa0p-5  ,
0x1.98800000000000p-5  , 0x1.a0b40000000050p-5  , 0x1.a90bffffffffb0p-5  , 0x1.b187ffffffffc0p-5  , 0x1.ba27ffffffffc0p-5  , 0x1.c2edffffffffa0p-5  , 0x1.cbda0000000030p-5  , 0x1.d4e9ffffffffb0p-5  ,
0x1.de21fffffffff0p-5  , 0x1.e7800000000000p-5  , 0x1.f105ffffffffd0p-5  , 0x1.fab20000000060p-5  , 0x1.0243ffffffffe0p-4  , 0x1.0742fffffffff0p-4  , 0x1.0c55ffffffffd0p-4  , 0x1.117e0000000010p-4  ,
0x1.16bb0000000020p-4  , 0x1.1c0d0000000010p-4  , 0x1.2174ffffffffe0p-4  , 0x1.26f2fffffffff0p-4  , 0x1.2c86ffffffffd0p-4  , 0x1.32310000000000p-4  , 0x1.37f10000000000p-4  , 0x1.3dc7ffffffffc0p-4  ,
0x1.43b5ffffffffd0p-4  , 0x1.49bb0000000020p-4  , 0x1.4fd80000000040p-4  , 0x1.560c0000000020p-4  , 0x1.5c580000000040p-4  , 0x1.62bc0000000020p-4  , 0x1.69390000000030p-4  , 0x1.6fce0000000010p-4  ,
0x1.767c0000000020p-4  , 0x1.7d42fffffffff0p-4  , 0x1.8423ffffffffe0p-4  , 0x1.8b1e0000000010p-4  , 0x1.9231fffffffff0p-4  , 0x1.99600000000000p-4  , 0x1.a0a90000000190p-4  , 0x1.a80c0000000090p-4  ,
0x1.af89ffffffffb0p-4  , 0x1.b722ffffffff00p-4  , 0x1.bed6fffffffe70p-4  , 0x1.c6a700000000b0p-4  , 0x1.ce930000000140p-4  , 0x1.d69b0000000020p-4  , 0x1.debf00000001d0p-4  , 0x1.e6ff00000001d0p-4  ,
0x1.ef5bfffffffe50p-4  , 0x1.f7d60000000050p-4  , 0x1.00367fffffff60p-3  , 0x1.04907fffffff10p-3  , 0x1.08f980000000a0p-3  , 0x1.0d717fffffff80p-3  , 0x1.11f88000000030p-3  , 0x1.168e8000000080p-3  ,
0x1.1b338000000060p-3  , 0x1.1fe7fffffffee0p-3  , 0x1.24ac0000000090p-3  , 0x1.297f80000000f0p-3  , 0x1.2e627ffffffff0p-3  , 0x1.33557ffffffef0p-3  , 0x1.38588000000030p-3  , 0x1.3d6b7fffffff40p-3  ,
0x1.428e8000000080p-3  , 0x1.47c17fffffff80p-3  , 0x1.4d0480000000c0p-3  , 0x1.52580000000120p-3  , 0x1.57bc0000000090p-3  , 0x1.5d307fffffff10p-3  , 0x1.62b57ffffffef0p-3  , 0x1.684b7fffffff40p-3  ,
0x1.6df200000000e0p-3  , 0x1.73a980000000a0p-3  , 0x1.797200000000e0p-3  , 0x1.7f4b7fffffff40p-3  , 0x1.85367fffffff60p-3  , 0x1.8b327ffffffff0p-3  , 0x1.91400000000000p-3  , 0x1.975e8000000080p-3  ,
0x1.9d8e8000000080p-3  , 0x1.a3d00000000000p-3  , 0x1.aa22ffffffff00p-3  , 0x1.b087fffffffee0p-3  , 0x1.b6fe8000000080p-3  , 0x1.bd8700000000b0p-3  , 0x1.c4210000000070p-3  , 0x1.cacd0000000100p-3  ,
0x1.d18b0000000020p-3  , 0x1.d85b0000000020p-3  , 0x1.df3d0000000100p-3  , 0x1.e6310000000070p-3  , 0x1.ed3700000000b0p-3  , 0x1.f44f80000000f0p-3  , 0x1.fb79ffffffffb0p-3  , 0x1.015b8000000060p-2  ,
0x1.05030000000020p-2  , 0x1.08b3bfffffffe0p-2  , 0x1.0c6d8000000010p-2  , 0x1.10308000000030p-2  , 0x1.13fcc000000050p-2  , 0x1.17d1ffffffffb0p-2  , 0x1.1bb08000000030p-2  , 0x1.1f983fffffff90p-2  ,
0x1.23893ffffffff0p-2  , 0x1.27838000000060p-2  , 0x1.2b86ffffffff90p-2  , 0x1.2f938000000060p-2  , 0x1.33a93ffffffff0p-2  , 0x1.37c83fffffff90p-2  , 0x1.3bf08000000030p-2  , 0x1.4021ffffffffb0p-2  ,
0x1.445c7fffffffa0p-2  , 0x1.48a03fffffff90p-2  , 0x1.4ced4000000080p-2  , 0x1.51433fffffffb0p-2  , 0x1.55a27ffffffff0p-2  , 0x1.5a0abfffffff80p-2  , 0x1.5e7c0000000090p-2  , 0x1.62f68000000080p-2  ,
0x1.6779ffffffffb0p-2  , 0x1.6c068000000080p-2  , 0x1.709c0000000090p-2  , 0x1.753a7ffffffff0p-2  , 0x1.79e1c000000030p-2  , 0x1.7e91c000000030p-2  , 0x1.834abfffffff80p-2  , 0x1.880c7fffffffa0p-2  ,
0x1.8cd6ffffffff90p-2  , 0x1.91aa4000000060p-2  , 0x1.96863fffffffd0p-2  , 0x1.9b6b0000000020p-2  , 0x1.a0583fffffff90p-2  , 0x1.a54e0000000050p-2  , 0x1.aa4c4000000020p-2  , 0x1.af52bfffffff80p-2  ,
0x1.b461c000000030p-2  , 0x1.b9790000000070p-2  , 0x1.be988000000030p-2  , 0x1.c3c03fffffff90p-2  , 0x1.c8f00000000000p-2  , 0x1.ce27c000000070p-2  , 0x1.d3677fffffffd0p-2  , 0x1.d8aeffffffff90p-2  ,
0x1.ddfe3fffffffd0p-2  , 0x1.e3554000000080p-2  , 0x1.e8b3ffffffff70p-2  , 0x1.ee1a4000000060p-2  , 0x1.f3880000000000p-2  , 0x1.f8fcffffffffe0p-2  , 0x1.fe797fffffff80p-2  , 0x1.01ff7fffffffd0p-1  ,
0x1.04c9dffffffff0p-1  , 0x1.079c0000000000p-1  , 0x1.0a75dffffffff0p-1  , 0x1.0d57a000000020p-1  , 0x1.10415fffffffc0p-1  , 0x1.13334000000040p-1  , 0x1.162d3ffffffff0p-1  , 0x1.192f6000000000p-1  ,
0x1.1c39dffffffff0p-1  , 0x1.1f4cbfffffffc0p-1  , 0x1.22684000000020p-1  , 0x1.258c5fffffffe0p-1  , 0x1.28b92000000030p-1  , 0x1.2beec000000010p-1  , 0x1.2f2d3ffffffff0p-1  , 0x1.3274bfffffffc0p-1  ,
0x1.35c53ffffffff0p-1  , 0x1.391f0000000020p-1  , 0x1.3c81ffffffffb0p-1  , 0x1.3fee6000000030p-1  , 0x1.43644000000020p-1  , 0x1.46e3a000000020p-1  , 0x1.4a6cbfffffffc0p-1  , 0x1.4dffa000000020p-1  ,
0x1.519c5fffffffe0p-1  , 0x1.55434000000040p-1  , 0x1.58f41fffffffc0p-1  , 0x1.5caf4000000040p-1  , 0x1.6074a000000000p-1  , 0x1.64448000000030p-1  , 0x1.681edfffffffd0p-1  , 0x1.6c040000000000p-1  ,
0x1.6ff3e000000040p-1  , 0x1.73eea000000040p-1  , 0x1.77f48000000030p-1  , 0x1.7c055fffffffc0p-1  , 0x1.80219fffffffd0p-1  , 0x1.84493ffffffff0p-1  , 0x1.887c4000000020p-1  , 0x1.8cbb0000000020p-1  ,
0x1.91058000000010p-1  , 0x1.955be000000040p-1  , 0x1.99be3fffffffd0p-1  , 0x1.9e2cbfffffffc0p-1  , 0x1.a2a77fffffffd0p-1  , 0x1.a72ec000000010p-1  , 0x1.abc26000000030p-1  , 0x1.b062c000000010p-1  ,
0x1.b5100000000000p-1  , 0x1.b9ca3fffffffd0p-1  , 0x1.be918000000010p-1  , 0x1.c3660000000050p-1  , 0x1.c847e000000040p-1  , 0x1.cd374000000040p-1  , 0x1.d2344000000020p-1  , 0x1.d73f0000000020p-1  ,
0x1.dc57e000000040p-1  , 0x1.e17edfffffffd0p-1  , 0x1.e6b40000000000p-1  , 0x1.ebf7a000000020p-1  , 0x1.f149dffffffff0p-1  , 0x1.f6aadfffffffd0p-1  , 0x1.fc1ac000000010p-1  , 0x1.ffff0000000020p-1  ,
1}; // 1 extra breakpoint for the 3 points polynomial interpolation (not included in pre-lookup)


/**
 * @brief Lookup table with uneven spacing
 */
const float lut_uneven [513] = {
-0x1.00000000p+4 , -0x1.fd338120p+3 , -0x1.fa8ff970p+3 , -0x1.f810fa50p+3 , -0x1.f5b2c3dcp+3 , -0x1.f37222bcp+3 , -0x1.f14c5610p+3 , -0x1.ef3efb00p+3 ,
-0x1.ed47fcb8p+3 , -0x1.eb6587b4p+3 , -0x1.e995ff70p+3 , -0x1.e7d7f62cp+3 , -0x1.e62a2604p+3 , -0x1.e48b6b70p+3 , -0x1.e2fac094p+3 , -0x1.e177394cp+3 ,
-0x1.e0000000p+3 , -0x1.de9452c8p+3 , -0x1.dd338120p+3 , -0x1.dbdce9dcp+3 , -0x1.da8ff970p+3 , -0x1.d94c2874p+3 , -0x1.d810fa50p+3 , -0x1.d6ddfc2cp+3 ,
-0x1.d5b2c3dcp+3 , -0x1.d48eef18p+3 , -0x1.d37222bcp+3 , -0x1.d25c0a04p+3 , -0x1.d14c5610p+3 , -0x1.d042bd4cp+3 , -0x1.cf3efb00p+3 , -0x1.ce40cee0p+3 ,
-0x1.cd47fcb8p+3 , -0x1.cc544c04p+3 , -0x1.cb6587b4p+3 , -0x1.ca7b7dd8p+3 , -0x1.c995ff70p+3 , -0x1.c8b4e028p+3 , -0x1.c7d7f62cp+3 , -0x1.c6ff19e8p+3 ,
-0x1.c62a2604p+3 , -0x1.c55731b4p+3 , -0x1.c3bf9f9cp+3 , -0x1.c235a420p+3 , -0x1.c0b85ebcp+3 , -0x1.bf470404p+3 , -0x1.bde0db0cp+3 , -0x1.bc853b34p+3 ,
-0x1.bb338a54p+3 , -0x1.b9eb3b20p+3 , -0x1.b8abcbc0p+3 , -0x1.b774c4a0p+3 , -0x1.b645b754p+3 , -0x1.b51e3dc4p+3 , -0x1.b3706f4cp+3 , -0x1.b1d1b658p+3 ,
-0x1.b0410d0cp+3 , -0x1.aebd8748p+3 , -0x1.ad464f74p+3 , -0x1.abdaa3a8p+3 , -0x1.aa79d360p+3 , -0x1.a9233d74p+3 , -0x1.a7d64e54p+3 , -0x1.a6927e9cp+3 ,
-0x1.a55751b4p+3 , -0x1.a42454c0p+3 , -0x1.a2970988p+3 , -0x1.a116aa78p+3 , -0x1.9fa2671cp+3 , -0x1.9e39821cp+3 , -0x1.9cdb4ef8p+3 , -0x1.9b873014p+3 ,
-0x1.9a3c9510p+3 , -0x1.98abebc0p+3 , -0x1.97286600p+3 , -0x1.95b12e28p+3 , -0x1.9445825cp+3 , -0x1.92e4b218p+3 , -0x1.918e1c28p+3 , -0x1.90412d0cp+3 ,
-0x1.8ebda748p+3 , -0x1.8d466f74p+3 , -0x1.8bdac3a8p+3 , -0x1.8a79f360p+3 , -0x1.89235d74p+3 , -0x1.879fd7b0p+3 , -0x1.86289fdcp+3 , -0x1.84bcf410p+3 ,
-0x1.835c23c8p+3 , -0x1.82058ddcp+3 , -0x1.8089d084p+3 , -0x1.7f19e5c8p+3 , -0x1.7db516dcp+3 , -0x1.7c5abd08p+3 , -0x1.7ae0daa0p+3 , -0x1.7972adc4p+3 ,
-0x1.780f8244p+3 , -0x1.76b6b3b0p+3 , -0x1.75430630p+3 , -0x1.73daae0cp+3 , -0x1.727cff8cp+3 , -0x1.7107ef10p+3 , -0x1.6f9e4948p+3 , -0x1.6e3f60a4p+3 ,
-0x1.6ccc16c8p+3 , -0x1.6b641c50p+3 , -0x1.6a06c604p+3 , -0x1.6897a03cp+3 , -0x1.67338a90p+3 , -0x1.65d9df50p+3 , -0x1.64709570p+3 , -0x1.63120350p+3 ,
-0x1.61a59820p+3 , -0x1.604413b8p+3 , -0x1.5ed64b98p+3 , -0x1.5d737ee0p+3 , -0x1.5c1b09a4p+3 , -0x1.5ab7ba64p+3 , -0x1.595eca24p+3 , -0x1.57fc3630p+3 ,
-0x1.56a3f678p+3 , -0x1.55432630p+3 , -0x1.53dace0cp+3 , -0x1.527d1f8cp+3 , -0x1.5118c364p+3 , -0x1.4fbed5b8p+3 , -0x1.4e5ef9e8p+3 , -0x1.4cf9fe00p+3 ,
-0x1.4b9f79c8p+3 , -0x1.4a4068b4p+3 , -0x1.48dd79e0p+3 , -0x1.4784e480p+3 , -0x1.4628dfdcp+3 , -0x1.44ca0030p+3 , -0x1.43753e90p+3 , -0x1.421df2a8p+3 ,
-0x1.40c499b8p+3 , -0x1.3f69a80cp+3 , -0x1.3e0d895cp+3 , -0x1.3cbb60c8p+3 , -0x1.3b682c34p+3 , -0x1.3a14463cp+3 , -0x1.38c00278p+3 , -0x1.376badd8p+3 ,
-0x1.36178ef4p+3 , -0x1.34c3e670p+3 , -0x1.3370ef4cp+3 , -0x1.321edf50p+3 , -0x1.30cde750p+3 , -0x1.2f7e3390p+3 , -0x1.2e2fec08p+3 , -0x1.2ce334b8p+3 ,
-0x1.2b90c348p+3 , -0x1.2a4088b4p+3 , -0x1.28f29960p+3 , -0x1.27a706fcp+3 , -0x1.26574160p+3 , -0x1.250a5240p+3 , -0x1.23c03f9cp+3 , -0x1.2272f778p+3 ,
-0x1.2128e820p+3 , -0x1.1fdc4d30p+3 , -0x1.1e933438p+3 , -0x1.1d4822a4p+3 , -0x1.1c00cb84p+3 , -0x1.1ab7fa64p+3 , -0x1.19730dc4p+3 , -0x1.182d13b8p+3 ,
-0x1.16e65e90p+3 , -0x1.159f3a80p+3 , -0x1.145c69ecp+3 , -0x1.13197370p+3 , -0x1.11d69548p+3 , -0x1.109408c8p+3 , -0x1.0f52028cp+3 , -0x1.0e10b2d0p+3 ,
-0x1.0cd045a8p+3 , -0x1.0b90e348p+3 , -0x1.0a52b044p+3 , -0x1.0915cdccp+3 , -0x1.07d6ee54p+3 , -0x1.0699c6a4p+3 , -0x1.055e6ce0p+3 , -0x1.0421ccc4p+3 ,
-0x1.02e749f0p+3 , -0x1.01abf3a0p+3 , -0x1.0072fb2cp+3 , -0x1.fe732390p+2 , -0x1.fbffecb0p+2 , -0x1.f99227e8p+2 , -0x1.f72496e8p+2 , -0x1.f4b79f88p+2 ,
-0x1.f24b9fb0p+2 , -0x1.efe0ed98p+2 , -0x1.ed77d860p+2 , -0x1.eb10a890p+2 , -0x1.e8aba038p+2 , -0x1.e648fbb0p+2 , -0x1.e3e4afe8p+2 , -0x1.e1836790p+2 ,
-0x1.df254cf8p+2 , -0x1.dcc69528p+2 , -0x1.da67ae40p+2 , -0x1.d80cbcc0p+2 , -0x1.d5b23030p+2 , -0x1.d35861a8p+2 , -0x1.d0ffa358p+2 , -0x1.cea840d8p+2 ,
-0x1.cc527fb8p+2 , -0x1.c9fe9fb0p+2 , -0x1.c7acdb18p+2 , -0x1.c55d6728p+2 , -0x1.c30d78c0p+2 , -0x1.c0c05cf8p+2 , -0x1.be736390p+2 , -0x1.bc29a8d8p+2 ,
-0x1.b9e096a8p+2 , -0x1.b7987968p+2 , -0x1.b5519790p+2 , -0x1.b30c3220p+2 , -0x1.b0c884d0p+2 , -0x1.ae86c678p+2 , -0x1.ac44d4a8p+2 , -0x1.aa054e80p+2 ,
-0x1.a7c62458p+2 , -0x1.a589ce60p+2 , -0x1.a34e4f48p+2 , -0x1.a113ea88p+2 , -0x1.9edade70p+2 , -0x1.9ca36470p+2 , -0x1.9a6db158p+2 , -0x1.9839f5b0p+2 ,
-0x1.96068910p+2 , -0x1.93d57ef0p+2 , -0x1.91a53c48p+2 , -0x1.8f760130p+2 , -0x1.8d4808f8p+2 , -0x1.8b1b8a58p+2 , -0x1.88f0b7e0p+2 , -0x1.86c7bff0p+2 ,
-0x1.849f48e0p+2 , -0x1.82790ff0p+2 , -0x1.8053c578p+2 , -0x1.7e2fa2c0p+2 , -0x1.7c0e3ea8p+2 , -0x1.79ecfe58p+2 , -0x1.77cd7920p+2 , -0x1.75afd888p+2 ,
-0x1.7392ffe8p+2 , -0x1.71786390p+2 , -0x1.6f5eee70p+2 , -0x1.6d46d1c8p+2 , -0x1.6b303b38p+2 , -0x1.691b54d8p+2 , -0x1.67072b98p+2 , -0x1.64f50970p+2 ,
-0x1.62e40128p+2 , -0x1.60d44208p+2 , -0x1.5ec5f7e8p+2 , -0x1.5cb94b40p+2 , -0x1.5aae6180p+2 , -0x1.58a46be0p+2 , -0x1.569c85b0p+2 , -0x1.5495e4c8p+2 ,
-0x1.5290b248p+2 , -0x1.508d1448p+2 , -0x1.4e8a55c0p+2 , -0x1.4c897890p+2 , -0x1.4a89cb40p+2 , -0x1.488b7628p+2 , -0x1.468e9e98p+2 , -0x1.44936730p+2 ,
-0x1.429931f8p+2 , -0x1.40a0e260p+2 , -0x1.3ea9dd28p+2 , -0x1.3cb44660p+2 , -0x1.3ac03f80p+2 , -0x1.38cd3c90p+2 , -0x1.36dc0c10p+2 , -0x1.34ec2448p+2 ,
-0x1.32fda740p+2 , -0x1.3110b4b0p+2 , -0x1.2f24cff0p+2 , -0x1.2d3ab448p+2 , -0x1.2b51e6d0p+2 , -0x1.296a8750p+2 , -0x1.2784b348p+2 , -0x1.259ffb38p+2 ,
-0x1.23bd08c0p+2 , -0x1.21db6dc0p+2 , -0x1.1ffb4788p+2 , -0x1.1e1cb138p+2 , -0x1.1c3f4698p+2 , -0x1.1a63a150p+2 , -0x1.18895e30p+2 , -0x1.16b09800p+2 ,
-0x1.14d8f3d8p+2 , -0x1.130300f0p+2 , -0x1.112e6520p+2 , -0x1.0f5b3a68p+2 , -0x1.0d8998e0p+2 , -0x1.0bb92e10p+2 , -0x1.09ea14c0p+2 , -0x1.081ccaa8p+2 ,
-0x1.06509b60p+2 , -0x1.04860338p+2 , -0x1.02bcb700p+2 , -0x1.00f52bd8p+2 , -0x1.fe5d7720p+1 , -0x1.fad3b3d0p+1 , -0x1.f74d33d0p+1 , -0x1.f3c96db0p+1 ,
-0x1.f0488bd0p+1 , -0x1.eccab590p+1 , -0x1.e94f6c20p+1 , -0x1.e5d6da80p+1 , -0x1.e26128a0p+1 , -0x1.deee7bb0p+1 , -0x1.db7ef630p+1 , -0x1.d8122360p+1 ,
-0x1.d4a828f0p+1 , -0x1.d14129f0p+1 , -0x1.cddcba50p+1 , -0x1.ca7b89f0p+1 , -0x1.c71d2d10p+1 , -0x1.c3c1c480p+1 , -0x1.c068ec20p+1 , -0x1.bd1347c0p+1 ,
-0x1.b9c07360p+1 , -0x1.b6708dc0p+1 , -0x1.b32339c0p+1 , -0x1.afd91010p+1 , -0x1.ac91b390p+1 , -0x1.a94d4110p+1 , -0x1.a60b61d0p+1 , -0x1.a2cca400p+1 ,
-0x1.9f90b0e0p+1 , -0x1.9c57a340p+1 , -0x1.992193e0p+1 , -0x1.95ee31e0p+1 , -0x1.92bd9800p+1 , -0x1.8f8fdf00p+1 , -0x1.8c651df0p+1 , -0x1.893d6a40p+1 ,
-0x1.86187840p+1 , -0x1.82f65f10p+1 , -0x1.7fd73420p+1 , -0x1.7cbb0b60p+1 , -0x1.79a19e10p+1 , -0x1.768b01a0p+1 , -0x1.737749f0p+1 , -0x1.70668980p+1 ,
-0x1.6d58d190p+1 , -0x1.6a4ddfa0p+1 , -0x1.6745c670p+1 , -0x1.64409730p+1 , -0x1.613e61c0p+1 , -0x1.5e3ee810p+1 , -0x1.5b423b50p+1 , -0x1.58486ba0p+1 ,
-0x1.555187c0p+1 , -0x1.525d9d70p+1 , -0x1.4f6cb960p+1 , -0x1.4c7ea140p+1 , -0x1.49936310p+1 , -0x1.46ab0b90p+1 , -0x1.43c5a6c0p+1 , -0x1.40e2fdf0p+1 ,
-0x1.3e035eb0p+1 , -0x1.3b269280p+1 , -0x1.384ca490p+1 , -0x1.35759f40p+1 , -0x1.32a14f30p+1 , -0x1.2fcffba0p+1 , -0x1.2d017170p+1 , -0x1.2a35f470p+1 ,
-0x1.276d51c0p+1 , -0x1.24a791b0p+1 , -0x1.21e4bbc0p+1 , -0x1.1f24a010p+1 , -0x1.1c677d20p+1 , -0x1.19ad2360p+1 , -0x1.16f5ce50p+1 , -0x1.14414ec0p+1 ,
-0x1.118faac0p+1 , -0x1.0ee0e7c0p+1 , -0x1.0c350a80p+1 , -0x1.098c1780p+1 , -0x1.06e61290p+1 , -0x1.0442cfc0p+1 , -0x1.01a28280p+1 , -0x1.fe09fee0p+0 ,
-0x1.f8d4ef80p+0 , -0x1.f3a580c0p+0 , -0x1.ee7c10e0p+0 , -0x1.e9584ac0p+0 , -0x1.e43a3300p+0 , -0x1.df2221c0p+0 , -0x1.da0fc3a0p+0 , -0x1.d5031b00p+0 ,
-0x1.cffc29c0p+0 , -0x1.cafaf140p+0 , -0x1.c5ff7220p+0 , -0x1.c109fa40p+0 , -0x1.bc1a3960p+0 , -0x1.b7302ea0p+0 , -0x1.b24bd8a0p+0 , -0x1.ad6d35c0p+0 ,
-0x1.a8948ca0p+0 , -0x1.a3c18fe0p+0 , -0x1.9ef43c80p+0 , -0x1.9a2cd580p+0 , -0x1.956b0f60p+0 , -0x1.90af2a60p+0 , -0x1.8bf91fc0p+0 , -0x1.8748a620p+0 ,
-0x1.829dfa80p+0 , -0x1.7df91560p+0 , -0x1.7959ef80p+0 , -0x1.74c08100p+0 , -0x1.702d00e0p+0 , -0x1.6b9f64a0p+0 , -0x1.67176540p+0 , -0x1.629536e0p+0 ,
-0x1.5e18cf40p+0 , -0x1.59a22400p+0 , -0x1.55312ac0p+0 , -0x1.50c5d960p+0 , -0x1.4c605e20p+0 , -0x1.4800acc0p+0 , -0x1.43a6b900p+0 , -0x1.3f52ada0p+0 ,
-0x1.3b044680p+0 , -0x1.36bbad20p+0 , -0x1.3278d3c0p+0 , -0x1.2e3bad00p+0 , -0x1.2a045f20p+0 , -0x1.25d2db20p+0 , -0x1.21a71240p+0 , -0x1.1d8127c0p+0 ,
-0x1.19610bc0p+0 , -0x1.1546ae80p+0 , -0x1.11320020p+0 , -0x1.0d232180p+0 , -0x1.091a01a0p+0 , -0x1.0516bf40p+0 , -0x1.01191a40p+0 , -0x1.fa3fda80p-1 ,
-0x1.f24d8c40p-1 , -0x1.ea5b2d40p-1 , -0x1.e268f980p-1 , -0x1.da76d340p-1 , -0x1.d2849e40p-1 , -0x1.ca924080p-1 , -0x1.c29ff680p-1 , -0x1.baadfb40p-1 ,
-0x1.b2bbe0c0p-1 , -0x1.aac9e3c0p-1 , -0x1.a2d79b80p-1 , -0x1.9ae545c0p-1 , -0x1.92f31e40p-1 , -0x1.8b00c140p-1 , -0x1.830e6bc0p-1 , -0x1.7b1c0bc0p-1 ,
-0x1.7329dcc0p-1 , -0x1.6b378180p-1 , -0x1.63453700p-1 , -0x1.5b52ee00p-1 , -0x1.53609880p-1 , -0x1.4b6e7280p-1 , -0x1.437c2640p-1 , -0x1.3b89f0c0p-1 ,
-0x1.3397c680p-1 , -0x1.2ba557c0p-1 , -0x1.23b32680p-1 , -0x1.1bc0e480p-1 , -0x1.13cecdc0p-1 , -0x1.0bdc9780p-1 , -0x1.03ea7e00p-1 , -0x1.f7f07300p-2 ,
-0x1.e80c0e00p-2 , -0x1.d827c300p-2 , -0x1.c8430c00p-2 , -0x1.b85ee080p-2 , -0x1.a87a4080p-2 , -0x1.9895a680p-2 , -0x1.88b18900p-2 , -0x1.78ccf580p-2 ,
-0x1.68e86580p-2 , -0x1.5903dc00p-2 , -0x1.491f5b00p-2 , -0x1.393ae780p-2 , -0x1.29568600p-2 , -0x1.1971cd80p-2 , -0x1.098da600p-2 , -0x1.f3527300p-3 ,
-0x1.d3892900p-3 , -0x1.b3bf8800p-3 , -0x1.93f68200p-3 , -0x1.742d5b00p-3 , -0x1.54643200p-3 , -0x1.349b2700p-3 , -0x1.14d25d00p-3 , -0x1.ea13ea00p-4 ,
-0x1.aa810c00p-4 , -0x1.6aee0c00p-4 , -0x1.2b5cc400p-4 , -0x1.d7950400p-5 , -0x1.586f5400p-5 , -0x1.b2928800p-6 , -0x1.688eb000p-7 , 0x1.1d000000p-18 ,
0}; // 1 extra lut value for the 3 points polynomial interpolation (not included in lookup)



// HELPER FUNCTIONS


/**
 * @brief same as abs() 
 */
float absolute(float x) {
    return (x < 0)? -x : x;
}


/**
 * @brief Comparator function used in qsort.
 */
int cmp_floats_ascending(const void* a, const void* b) {
  if (*(float*)a > *(float*)b)
    return 1;
  if (*(float*)a < *(float*)b)
    return -1;
  return 0;
}


/**
 * @brief Efficient method to horizontally adds all single-precision (32-bit) floating-point elements in mm and pack the results in a float.
 * Implemented by Peter Cordes (source: https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction)
 */
float sum_sse(__m128 mm) { // mm = [D|C|B|A]
    __m128 shuf = _mm_movehdup_ps(mm); // duplicate elements 3,1 to 2,0: shuf = [D|D|B|B]
    __m128 sums = _mm_add_ps(mm, shuf); // sums = [D+D|C+D|B+B|A+B]
    shuf = _mm_movehl_ps(shuf, sums); // move the upper 2 elements of sums to the lower 2 elements of shuf: shuf = shuf = [D+D|C+D|D+D|C+D]
    sums = _mm_add_ss(sums, shuf); // sums = [...|...|...|A+B+C+D] 
    return _mm_cvtss_f32(sums); // extract the lower element A+B+C+D of sums
}



// INTEPROLATION FUNCTIONS


/**
 * @brief Returns the result f(x) of the linear intepolation of the function f described by 2 points
 */
float linear_interp_2(float x1,float x2,float y1,float y2, float x) {
    return (x-x1)*(y2-y1)/(x2-x1) + y1; // absorption through addition/substraction was "avoided"
}


/**
 * @brief Returns the result f(x) of the polynomial intepolation of the function f described by 3 points,
 * using the  barycentric form of the Lagrange formula for a better performance.
 * a, b, c are used to calculate the barycentric weights w1=1/(a*b) w2-=1/(a*c) w3=1/(b*c), 
 * assuming that: x1!=x2 and x1!=x3 and x2!x3, which was already verified programatically through a loop through the pre-lookup table (uneven lut).
 * d1, d2, and d3 are used to calculate the improved lagrange polynomial l=d1*d2*d3, assuming that x!=x1 and x!=x2 and x!x3, 
 * which should be verified by the caller method, which has the advantage of verifying only 2 inequalitie since x is "in" an interval of the pre-lookup table. 
 */
float lagrange_interp_3(float x1,float x2,float x3,float y1,float y2,float y3, float x) {
    float a, b, c, d1, d2, d3; 
    a = x2 - x1;
    b = x3 - x1;
    c = x3 - x2;
    d1 = x - x1;
    d2 = x - x2;
    d3 = x - x3;
    return d1*d2*d3 * (y1/(a*b*d1) + y3/(b*c*d3) - y2/(a*c*d2)); // barycentric form of the Lagrange formula p(x) = l * (w1*y1/d1 + w2*y2/d2 + w3*y3/d3)
}



// LOG2 FUNCTIONS


/**
 * @brief log2 aproximation with even lut, linear inteprolation, and a heuristic adjustement c 
 * corresponding to the negation of the mean relative deviation from the gcc implementation of log2
 * measured with 1 million values of x in [0,1].
 * The inital lut has 512=2^9=0x200 entries for x in [2^(-9) , 1-2^(-9)]. Casting return the nearest lower int, which should be i1 but is i2 
 * since the array indexing begins with 0 and not 1
 */
float log2_lut_even_linear(float x) {
    int i1, i2;
    float u, x1, x2, c;
    u = 0x0.008p+0; // 2^(-9)
    i2 = (int) (x * 0x200p+0); // "left shift" of 9
    i1 = i2 - 1;
    x1 = i1 * u;
    x2 = i2 * u;
    if (x == x1)
        return lut_even[i1];
    if (x == x2)
        return lut_even[i2]; // TODO: not optimal...
    c = -0.011149978265166283; // established heuristically
    return linear_interp_2(x1, x2, lut_even[i1], lut_even[i2], x) + c;
}


/**
 * @brief log2 aproximation with even lut, polynomial inteprolation, and a heuristic adjustement c 
 * corresponding to the negation of the mean relative deviation from the gcc implementation of log2 measured with 1 million values of x in [0,1].
 * The inital lut has 512=2^9=0x200 entries for x in [2^(-9) , 1-2^(-9)]. Casting return the nearest lower int, which should be i1 but is i2 
 * since the array indexing begins with 0 and not 1 
 */
float log2_lut_even_polynomial(float x) {
    int i1, i2, i3;
    float u, x1, x2, x3, y1, y2, y3, c;
    u = 0x0.008p+0; // 2^(-9) TODO: define u as a macro
    i2 = (int) (x * 0x200p+0); // "left shift" of 9
    i1 = i2 - 1;
    x1 = i1 * u;
    x2 = i2 * u;
    y1 = lut_even[i1];
    y2 = lut_even[i2];
    if (x==x1)
        return y1;
    if (x==x2)
        return y2; // TODO: not optimal...
    i3 = i2 + 1;
    x3 = i3 * u;
    y3 = lut_even[i3];
    c = -0.0167583879083395; // established heuristically
    return lagrange_interp_3(x1, x2, x3, y1, y2, y3, x) + c;
}


/**
 * @brief log2 aproximation with uneven lut, linear inteprolation, and a heuristic adjustement c 
 * corresponding to the negation of the mean relative deviation from the gcc implementation of log2 
 * measured with 1 million values of x in [0,1].  
 */
float log2_lut_uneven_linear(float x) {
    int i1, i2, i_mid;
    float x1, x2, c;
    // pre-lookup: find the interval [x1, x2] with indexes [i1,i2] that contains x in the breakpoint table using binary search (9 iterations)
    i1 = 0;
    i2 = 512;
    for (int n=0; n <= 8; n++) {
        i_mid = (int)(i1+i2)/2;
        if (x>=bps_uneven[i_mid])
            i1 = i_mid;
        else
            i2 = i_mid;
    }
    x1 = bps_uneven[i1];
    x2 = bps_uneven[i2];
    // TODO: not optimal... change this:
    if (x == x1)
        return lut_uneven[i1];
    if (x == x2)
        return lut_uneven[i2];
    c = -0.000016554402463953011; // established heuristically
    return linear_interp_2(x1, x2, lut_uneven[i1], lut_uneven[i2], x) + c;
}


/**
 * @brief log2 aproximation with uneven lut, polynomial inteprolation, and a heuristic adjustement c 
 * corresponding to the negation of the mean relative deviation from the gcc implementation of log2
 * measured with 1 million values of x in [0,1].
 */
float log2_lut_uneven_polynomial(float x) {
    int i1, i2, i_mid, i3;
    float x1, x2, x3, y1, y2, y3, c;
    // pre-lookup: find the interval [x1, x2] with indexes [i1,i2] that contains x in the breakpoint table with binary search (9 iterations)
    i1 = 0;
    i2 = 512;
    for (int n=0; n < 9; n++) {
        i_mid = (int)(i1+i2)/2;
        if (x>=bps_uneven[i_mid])
            i1 = i_mid;
        else
            i2 = i_mid;
    }
    x1 = bps_uneven[i1];
    x2 = bps_uneven[i2];
    y1 = lut_uneven[i1];
    y2 = lut_uneven[i2];
    // optional, TODO: test if it impacts precision and performace
    if (x == x1)
        return y1;
    if (x == x2)
        return y2; // TODO: not optimal...
    i3 = i2 + 1;
    x3 = bps_uneven[i3];
    y3 = lut_uneven[i3];
    c = -0.000019405963030294515; // established heuristically
    return lagrange_interp_3(x1, x2, x3, y1, y2, y3, x) + c;
}

void prepare(size_t len, const float data[len], float sorted_data[len]){
    // edge cases
    if (len <= 0){
        fprintf(stderr, "Fehler bei der Entropieberechnung! Die Arraylänge ist kleiner als 1\n");
        exit(EXIT_FAILURE);
    }
    if (len == 1){
        if(data[0] < 0.9 || data[0] > 1.1){
            fprintf(stderr,"Summe der Wahrscheinlichkeiten ist ungleich 1! Eingabedatei auf Korrektheit überprüfen!\n");
            exit(EXIT_FAILURE);
        }
        printf("Die berechnete Entropie beträgt: 0\n");
        exit(EXIT_SUCCESS);
    }
    // sort to counter the absorption problem causing precision loss when adding values
    memcpy(sorted_data, data, len*sizeof(float));
    // TODO: implement own function void sort_ascend(float *, size_t) and replace qsort
    qsort(sorted_data, len, sizeof(float), cmp_floats_ascending);
    // verify the summing property of the probability distribution
    float sum = 0; 
    for (size_t i=0; i<len; i++){
        sum+=sorted_data[i];
    }
    if (sum < 0.9 || sum > 1.1){ // due to absorption problem
        fprintf(stderr, "Summe der Wahrscheinlichkeiten ist ungleich 1! Eingabedatei auf Korrektheit überprüfen!\n");
        exit(EXIT_FAILURE);
    }
    return;
}

// ENTROPY 

/**
 * @brief Returns the entropy of the probabilty ditribution data[len] wheen it meets the requirements, or else -1.
 */
float entropy (size_t len, const float data[len]) {
    float sorted_data[len];
    prepare(len, data, sorted_data);
    // calculate entropy
    float p;
    float e = 0;
    for (size_t i=1; i<=len; i++) {
        // substract smaller values first to avoid the absorption precision problem 
        p = sorted_data[len-i];
        e -= p * log2_lut_uneven_polynomial(p);
    }
    return e;
}
//log2_lut_uneven_linear
float entropy_V5 (size_t len, const float data[len]) {
    float sorted_data[len];
    prepare(len, data, sorted_data);
    // calculate entropy
    float p;
    float e = 0;
    for (size_t i=1; i<=len; i++) {
        // substract smaller values first to avoid the absorption precision problem 
        p = sorted_data[len-i];
        e -= p * log2_lut_uneven_linear(p);
    }
    return e;
}

//even polynomial
float entropy_V6 (size_t len, const float data[len]) {
    float sorted_data[len];
    prepare(len, data, sorted_data);
    // calculate entropy
    float p;
    float e = 0;
    for (size_t i=1; i<=len; i++) {
        // substract smaller values first to avoid the absorption precision problem 
        p = sorted_data[len-i];
        e -= p * log2_lut_even_polynomial(p);
    }
    return e;
}

//even linear
float entropy_V7 (size_t len, const float data[len]) {
    float sorted_data[len];
    prepare(len, data, sorted_data);
    // calculate entropy
    float p;
    float e = 0;
    for (size_t i=1; i<=len; i++) {
        // substract smaller values first to avoid the absorption precision problem 
        p = sorted_data[len-i];
        e -= p * log2_lut_even_linear(p);
    }
    return e;
}


// SIMD OPTIMIZED VERSION OF THE BEST FUNCTIONS (>=SSE4)


/**
 * @brief Optimized with SSE: float lagrange_interp_3(float x1,float x2,float x3,float y1,float y2,float y3, float x)
 */
__m128 lagrange_interp_3_simd(__m128 x1,__m128 x2,__m128 x3,__m128 y1,__m128 y2,__m128 y3, __m128 x) {
    __m128 a = _mm_sub_ps(x2, x1);
    __m128 b = _mm_sub_ps(x3, x1);
    __m128 c = _mm_sub_ps(x3, x2);
    __m128 d1 = _mm_sub_ps(x, x1);
    __m128 d2 = _mm_sub_ps(x, x2);
    __m128 d3 = _mm_sub_ps(x, x3);
    __m128 l = _mm_mul_ps(_mm_mul_ps(d1,d2),d3);
    __m128 m1 = _mm_div_ps(y1,_mm_mul_ps(_mm_mul_ps(a,b),d1));
    __m128 m2 = _mm_div_ps(y2,_mm_mul_ps(_mm_mul_ps(a,c),d2));
    __m128 m3 = _mm_div_ps(y3,_mm_mul_ps(_mm_mul_ps(b,c),d3));
    return _mm_mul_ps(l,_mm_sub_ps(_mm_add_ps(m1,m3), m2));
}

/**
 * @brief Optimized with SSE: float log2_lut_uneven_linear(float x)
 */
__m128 log2_lut_uneven_polynomial_simd(__m128 x) {
    // pre-lookup: find the intervals [x1, x2] with indexes [i1,i2] that contains the x values in the breakpoint table with binary search (9 iterations)
    __m128i i1 = _mm_setzero_si128();
    __m128i i2 = _mm_set1_epi32(512);
    __m128i i_mid;
    __m128 mid_bps, mask, i1f, i2f, i_midf;
    for (int n=0; n < 9; n++) {
        i_mid = _mm_srli_epi32 (_mm_add_epi32(i1, i2), 1);
        // the only reasonable possibilty I found to gather elements of an array with unconsecutive indexes in sse4 is to serialise it like the following...
        // with avx, it would be via the intrinsic: __m256i _mm256_i32gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
        // but we are not using avx for portability reasons
        mid_bps = _mm_set_ps(bps_uneven[_mm_extract_epi32 (i_mid, 3)], bps_uneven[_mm_extract_epi32 (i_mid, 2)], 
                                bps_uneven[_mm_extract_epi32 (i_mid, 1)], bps_uneven[_mm_extract_epi32 (i_mid, 0)]);
        mask = _mm_cmpge_ps(x, mid_bps); // x element >= mid_bps element ? 0xFFFFFFFF : 0
        // get packed single precision version of i1, i2, i_mid for logical operations with i_mid (cannot be converted to epi32)
        i1f = _mm_cvtepi32_ps(i1);
        i2f = _mm_cvtepi32_ps(i2);
        i_midf = _mm_cvtepi32_ps(i_mid);
        // new indexes
        i1f = _mm_or_ps(_mm_andnot_ps(mask, i1f), _mm_and_ps (mask, i_midf)); // i1 = (not(mask) and i1) or (mask and i_mid)
        i1 = _mm_cvtps_epi32(i1f);
        i2f = _mm_or_ps(_mm_and_ps(mask, i2f),  _mm_andnot_ps(mask, i_midf)); // i2 = (not(mask) and i_mid) or (mask and i2)
        i2 = _mm_cvtps_epi32(i2f);
    }
    __m128i ones = _mm_set1_epi32(1);
    __m128i i3 = _mm_add_epi32(i2, ones);
    // serializing like the following is not optimal (see previous comment) ...
    __m128 x1 = _mm_set_ps(bps_uneven[_mm_extract_epi32 (i1, 3)], bps_uneven[_mm_extract_epi32 (i1, 2)], 
                        bps_uneven[_mm_extract_epi32 (i1, 1)], bps_uneven[_mm_extract_epi32 (i1, 0)]);
    __m128  x2 = _mm_set_ps(bps_uneven[_mm_extract_epi32 (i2, 3)], bps_uneven[_mm_extract_epi32 (i2, 2)], 
                        bps_uneven[_mm_extract_epi32 (i2, 1)], bps_uneven[_mm_extract_epi32 (i2, 0)]);
    __m128 x3 = _mm_set_ps(bps_uneven[_mm_extract_epi32 (i3, 3)], bps_uneven[_mm_extract_epi32 (i3, 2)], 
                        bps_uneven[_mm_extract_epi32 (i3, 1)], bps_uneven[_mm_extract_epi32 (i3, 0)]);
    __m128 y1 = _mm_set_ps(lut_uneven[_mm_extract_epi32 (i1, 3)], lut_uneven[_mm_extract_epi32 (i1, 2)], 
                        lut_uneven[_mm_extract_epi32 (i1, 1)], lut_uneven[_mm_extract_epi32 (i1, 0)]);
    __m128 y2 = _mm_set_ps(lut_uneven[_mm_extract_epi32 (i2, 3)], lut_uneven[_mm_extract_epi32 (i2, 2)], 
                        lut_uneven[_mm_extract_epi32 (i2, 1)], lut_uneven[_mm_extract_epi32 (i2, 0)]);
    __m128 y3 = _mm_set_ps(lut_uneven[_mm_extract_epi32 (i3, 3)], lut_uneven[_mm_extract_epi32 (i3, 2)], 
                        lut_uneven[_mm_extract_epi32 (i3, 1)], lut_uneven[_mm_extract_epi32 (i3, 0)]);
    __m128 c = _mm_set1_ps(-0.000019405963030294515); // change intrinsic for float
    return _mm_add_ps(lagrange_interp_3_simd(x1, x2, x3, y1, y2, y3, x), c);
}

/**
 * @brief Optimized with SSE: float entropy (size_t len, const float data[len]) 
 */
float entropy_V3 (size_t len, const float data[len]) {
    if (len <= 0){
        fprintf(stderr, "Fehler bei der Entropieberechnung! Die Arraylänge ist kleiner als 1\n");
        exit(EXIT_FAILURE);        
    }
    if (len == 1){
        if(data[0] < 0.9 || data[0] > 1.1){
            fprintf(stderr,"Summe der Wahrscheinlichkeiten ist ungleich 1! Eingabedatei auf Korrektheit überprüfen!\n");
            exit(EXIT_FAILURE);
        }
        printf("Die berechnete Entropie beträgt: 0\n");
        exit(EXIT_SUCCESS);
    }
    float sorted_data[len];
    memcpy(sorted_data, data, len*sizeof(float));
    // parrallelizing sorting seems to be out of the scope of this project
    // TODO: implement void sort(float *, size_t) function to replace qsort
    qsort(sorted_data, len, sizeof(float), cmp_floats_ascending);
    // verify a property of the probability distribution
    __m128 p;
    size_t rest = len % 4; 
    __m128 s = _mm_setzero_ps();
    float s_scal = 0;
    if (len != rest) {
        for (size_t i=0; i < len-rest; i+=4) {
            p = _mm_set_ps(sorted_data[i+3], sorted_data[i+2], sorted_data[i+1], sorted_data[i]);
            s = _mm_add_ps(s, p);
        }
        s_scal = sum_sse(s);
    }
    //for (size_t i=len-rest; i<len; i++, s_scal+=sorted_data[i]);
    for (size_t i=len-rest; i<len; i++){
        s_scal+=sorted_data[i];
    }
    if (s_scal < 0.9 || s_scal > 1.1){
        fprintf(stderr, "Summe der Wahrscheinlichkeiten ist ungleich 1! Eingabedatei auf Korrektheitüberprüfen!\n");
        exit(EXIT_FAILURE);       
    } // due to absorption problem

    // calculate entropy
    __m128 e = _mm_setzero_ps();
    float e_scal = 0;
    if (len != rest) {
        for (size_t i = 0; i < len-rest ; i+=4) {
            p = _mm_set_ps(sorted_data[len-i-1], sorted_data[len-i-2], sorted_data[len-i-3], sorted_data[len-i-4]); 
            e = _mm_sub_ps(e, _mm_mul_ps(p, log2_lut_uneven_polynomial_simd(p))); // absorption problem avoided
        }
        e_scal = sum_sse(e);
    }
    // the following contains redundancies that will hopefully be removed  by the compiler
    //for (size_t i=0; i < rest; i++, e_scal-=sorted_data[rest-i-1]*log2_lut_uneven_polynomial(sorted_data[rest-i-1])); // take close attention to rest=0 edge case
    for (size_t i=0; i < rest; i++){
        e_scal-=sorted_data[rest-i-1]*log2_lut_uneven_polynomial(sorted_data[rest-i-1]);
    }
    return e_scal;
}



// PERFORMANCE AND PRECISION TESTS


/**
 * @brief Helper function printing the mean absolute deviation of our 4 lut implementations in comparision to the gcc log2 function
 */
void abs_diff_avg () {
    float sum = 0;
    for (float x=0.0000001; x<=1; x+=0.0000001, sum+=abs(log2_lut_uneven_linear(x)-log2f(x))) {}
    float avg = sum / 10000000;
    printf("log2_lut_uneven_linear:     %.6f\n", avg);

    sum = 0;
    for (float x=0.0000001; x<=1; x+=0.0000001, sum+=abs(log2_lut_uneven_polynomial(x)-log2f(x))) {}
    avg = sum / 10000000;
    printf("log2_lut_uneven_polynomial: %.6f\n", avg);

    sum = 0;
    for (float x=0.0000001; x<=1; x+=0.0000001, sum+=abs(log2_lut_even_linear(x)-log2f(x))) {}
    avg = sum / 10000000;
    printf("log2_lut_even_linear:       %.6f\n", avg);

    sum = 0;
    for (float x=0.0000001; x<=1; x+=0.0000001, sum+=abs(log2_lut_even_polynomial(x)-log2f(x))) {}
    avg = sum / 10000000;
    printf("log2_lut_even_polynomial:   %.6f\n", avg);
    return;
}


void exec_time () {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (float x=0.001; x<=1; x+=0.001, log2_lut_uneven_linear(x)) {}
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("log2_lut_uneven_linear:     %.16f\n", end.tv_sec-start.tv_sec+(1e-9)*(end.tv_nsec-start.tv_nsec));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (float x=0.001; x<=1; x+=0.001, log2_lut_uneven_polynomial(x)) {}
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("log2_lut_uneven_polynomial: %.16f\n", end.tv_sec-start.tv_sec+(1e-9)*(end.tv_nsec-start.tv_nsec));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (float x=0.001; x<=1; x+=0.001, log2_lut_even_linear(x)) {}
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("log2_lut_even_linear:       %.16f\n", end.tv_sec-start.tv_sec+(1e-9)*(end.tv_nsec-start.tv_nsec));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (float x=0.001; x<=1; x+=0.001, log2_lut_even_polynomial(x)) {}
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("log2_lut_even_polynomial:   %.16f\n", end.tv_sec-start.tv_sec+(1e-9)*(end.tv_nsec-start.tv_nsec));

    __m128 xxxx;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (float x=0.001; x<=1; x+=0.001) {
        xxxx = _mm_set1_ps(x); // the overhead
        log2_lut_uneven_polynomial_simd(xxxx);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    struct timespec ostart, oend; 
    clock_gettime(CLOCK_MONOTONIC, &ostart);
    for (float x=0.001; x<=1; x+=0.001) {
        xxxx = _mm_set1_ps(x);
    }
    clock_gettime(CLOCK_MONOTONIC, &oend);
    printf("log2_lut_uneven_polynomial simd: %.16f\n", end.tv_sec-start.tv_sec+(1e-9)*(end.tv_nsec-start.tv_nsec)
                                                        - (oend.tv_sec-ostart.tv_sec+(1e-9)*(oend.tv_nsec-ostart.tv_nsec)));

}


/**
 * @brief Prints the Entropy of a random integer generating function randf (for example the C Library function rand) passed as a parameter, over a number of generated values len,
 * and compares it with the maximum attainable entropy.
 * 
 * @param len 
 * @param randf 
 */
void rand_entropy (int len, int (*randf)(void)) {
    srand(time(NULL)); // initialisation, should only be called once
    // generate array of random integers with the chosen function randf
    int random [len];
    for (int i = 0; i<len; i++) {
        random[i] = randf();
    }
    // generate the probability distribution
    float data [len];
    qsort(random, len, sizeof(float), cmp_floats_ascending);
    float counter = 1;
    int prev = -1;
    int j = 0;
    for (int i = 0; i<len; i++) {
        if (random[i] == prev) {
            counter++;
            data[j] = counter / (float) len;
        } else {
            counter = 1;
            prev = random[i];
            j++;
            data[j] = counter / (float) len;
        }
    }
    float max_e = log2f(j+1); // we do not use our log2 function since it is hightly precise only for x in ]0,1] ...
    // the rest of the array is "empty" ...
    while (j<len) {
        data[j] = 0;
        j++;
    }
    // slightly modified entropy calculation
    float sorted_data[len];
    prepare(len, data, sorted_data);
    float p;
    float e = 0;
    for (size_t i=1; i<=len; i++) {
        p = sorted_data[len-i];
        if (p!=0) {
            e -= p * log2_lut_uneven_polynomial(p);
        }
    }
    // print results
    printf("Entropy of the random integer generator function:       %.16f\n", e);
    printf("Maximum attainable Entropy:                             %.16f\n", max_e);
}



// LUT AND INPUT GENERATORS


/**
 * @brief Helper function printing the used even lut of 512 values in hexadecimal form to the console
 */
void lut_even_generator() {
    int n=0;
    for (float x=0.001953125; x<=1; x+=0.001953125){
        printf("%.6a , ", log2f(x));
        n++;
        if (n==8){
            printf("\n");
            n=0;
        }
    }
    return;
}

/**
 * @brief Fills array data[len] with values in ]0,1] in accordance with the definition of probability distribution.
 */
void data_generator (float *data, int len) { 
    float t = 0;
    float p;
    while (t<=0) {
        t = 1;
        for (int i = 0; i < (len-1); i++) {
            p = 2*(float)rand()/(float)RAND_MAX/((float)len);
            data[i] = p;
            t -= p;
        }
    }
    data[len-1] = t;
    return;
}


/*int main() {
    rand_entropy (10000, rand);
}*/