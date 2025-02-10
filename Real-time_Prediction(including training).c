#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NUC100Series.h"

#define PLL_CLOCK       50000000

/******************************************************************
 * dataset format setting
 ******************************************************************/

#define train_data_num 224			//Total number of training data
#define test_data_num 56		//Total number of testing data

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/
#define input_length 3				//The number of input
#define HiddenNodes 20 						//The number of neurons in hidden layer
#define target_num 7						//The number of output

const float LearningRate = 0.01;						//Learning Rate
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float goal_acc = 	0.94;							//Target accuracy

// Create training dataset/output
float train_data_input[train_data_num][input_length] = {
{633, 951, 547},{613, 927, 531},{644, 968, 566},{651, 971, 565},{585, 897, 507},{572, 887, 500},{548, 846, 469},{520, 820, 457},
{556, 854, 479},{512, 811, 448},{558, 839, 460},{510, 790, 436},{501, 775, 417},{496, 766, 416},{534, 807, 445},{536, 818, 452},
{535, 835, 472},{576, 888, 508},{608, 929, 539},{540, 853, 478},{495, 811, 460},{510, 823, 466},{525, 831, 469},{496, 808, 445},
{562, 881, 500},{579, 897, 508},{586, 909, 526},{550, 865, 494},{530, 840, 477},{503, 787, 439},{500, 778, 426},{576, 850, 460},
	
{752, 842, 352},{753, 858, 346},{713, 832, 358},{682, 782, 328},{754, 819, 329},{753, 811, 324},{774, 840, 330},{783, 848, 337},
{797, 863, 341},{830, 908, 349},{832, 913, 357},{801, 874, 366},{783, 855, 373},{785, 871, 355},{804, 905, 405},{792, 930, 389},
{785, 895, 391},{743, 872, 425},{780, 902, 428},{924, 1044, 454},{996, 1168, 472},{997, 1184, 456},{986, 1247, 461},{882, 1245, 440},
{870, 1240, 441},{847, 1225, 434},{834, 1211, 429},{747, 1133, 413},{940, 1261, 445},{983, 1247, 452},{977, 1250, 456},{995, 1209, 455},

{877, 1606, 1061},{871, 1611, 1067},{881, 1608, 1071},{872, 1608, 1029},{895, 1615, 1053},{1072, 1745, 1210},{1142, 1676, 1266},{1114, 1607, 1237},
{1096, 1559, 1214},{1104, 1580, 1218},{1088, 1557, 1204},{1071, 1548, 1195},{1053, 1512, 1176},{1030, 1465, 1141},{1046, 1496, 1157},{1000, 1429, 1100},
{1004, 1434, 1112},{1006, 1421, 1114},{976, 1398, 1088},{991, 1417, 1101},{980, 1390, 1096},{960, 1354, 1074},{940, 1338, 1055},{936, 1327, 1040},
{984, 1384, 1082},{1047, 1481, 1158},{1053, 1497, 1169},{1076, 1526, 1186},{1008, 1428, 1108},{1025, 1447, 1131},{1003, 1426, 1110},{1049, 1495, 1163},
	
{1020, 1606, 1226},{1024, 1598, 1220},{974, 1517, 1175},{930, 1448, 1142},{902, 1422, 1144},{933, 1449, 1161},{1019, 1619, 1234},{1031, 1624, 1237},
{1043, 1689, 1249},{988, 1532, 1186},{994, 1578, 1201},{999, 1613, 1216},{1015, 1644, 1227},{1019, 1604, 1221},{1020, 1599, 1223},{1010, 1584, 1213},
{1034, 1637, 1242},{1034, 1691, 1255},{877, 1647, 1092},{795, 1583, 1003},{777, 1574, 1002},{782, 1576, 996},{798, 1581, 996},{798, 1575, 994},
{796, 1574, 991},{779, 1565, 972},{792, 1559, 957},{758, 1546, 995},{768, 1555, 1027},{754, 1555, 1057},{728, 1545, 1016},{710, 1528, 1000},

{1146, 1807, 1384},{1144, 1808, 1384},{1143, 1832, 1378},{1135, 1847, 1371},{1129, 1875, 1358},{1135, 1876, 1361},{1136, 1865, 1363},{1129, 1780, 1338},
{1112, 1738, 1314},{1099, 1708, 1306},{1124, 1752, 1345},{1135, 1789, 1369},{1128, 1767, 1350},{1137, 1835, 1352},{1128, 1839, 1352},{1118, 1854, 1349},
{1134, 1847, 1363},{1140, 1827, 1363},{1139, 1837, 1372},{1138, 1845, 1372},{1146, 1842, 1380},{1150, 1852, 1382},{1122, 1880, 1367},{1102, 1889, 1350},
{1123, 1878, 1369},{1152, 1841, 1378},{1146, 1816, 1373},{1132, 1775, 1362},{1054, 1622, 1249},{943, 1457, 1118},{1012, 1564, 1205},{1108, 1719, 1316},
	
{1121, 2056, 1121},{1090, 2039, 1100},{1118, 2064, 1124},{1125, 2064, 1127},{1134, 2095, 1193},{1120, 2109, 1225},{1122, 2122, 1269},{1132, 2121, 1262},
{1139, 2127, 1251},{1132, 2131, 1261},{1121, 2122, 1237},{1133, 2112, 1211},{1032, 2007, 1092},{1025, 2003, 1101},{1070, 2046, 1140},{1112, 2084, 1184},
{1133, 2110, 1226},{1124, 2120, 1254},{1123, 2116, 1271},{1113, 2116, 1283},{1046, 2088, 1332},{1403, 2124, 1419},{1406, 2055, 1371},{1291, 1840, 1221},
{1373, 1964, 1318},{1024, 2048, 1282},{1039, 2054, 1266},{1050, 2031, 1190},{1030, 2004, 1111},{1015, 1987, 1086},{1007, 1963, 1056},{1012, 1959, 1045},
	
{502, 1051, 378},{518, 1076, 377},{528, 1060, 376},{520, 1063, 382},{529, 1077, 392},{521, 1074, 394},{524, 1063, 386},{533, 1076, 389},
{536, 1073, 391},{573, 1141, 411},{669, 940, 406},{495, 747, 382},{534, 769, 379},{643, 918, 397},{766, 1091, 447},{797, 1208, 478},
{601, 1125, 416},{646, 1177, 441},{576, 1098, 422},{593, 1110, 430},{723, 1234, 470},{691, 1208, 448},{712, 1217, 444},{678, 1188, 423},
{723, 1226, 442},{806, 1199, 480},{774, 1221, 485},{782, 1192, 470},{776, 1204, 472},{802, 1189, 481},{812, 1221, 493},{780, 1228, 481}
};

int train_data_output[train_data_num][target_num]  = {
{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},
{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},
{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},
{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},
	
{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},
{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},
{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},
{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},

{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},
{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},
{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},
{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},
	
{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},
{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},
{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},
{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},
	
{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},
{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},
{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},
{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},
	
{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},
{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},
{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},
{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},
	
{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},
{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},
{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},
{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1}
};

// Create testing dataset/output
float test_data_input[test_data_num][input_length] = {
{595, 907, 523},{605, 908, 514},{635, 943, 546},{645, 962, 555},{593, 898, 512},{576, 889, 508},{596, 914, 523},{616, 931, 537},
{955, 1099, 455},{918, 1035, 437},{851, 973, 425},{857, 975, 436},{931, 1073, 446},{924, 1074, 452},{861, 978, 433},{859, 963, 406},
{1040, 1478, 1156},{908, 1307, 1009},{848, 1222, 939},{975, 1394, 1084},{1056, 1498, 1172},{1129, 1686, 1259},{1104, 1742, 1247},{891, 1613, 1076},
{741, 1530, 981},{753, 1549, 984},{773, 1556, 976},{776, 1562, 1015},{789, 1569, 1000},{795, 1567, 1010},{1049, 1691, 1259},{1039, 1661, 1255},
{1144, 1812, 1367},{1108, 1897, 1380},{1000, 1852, 1305},{861, 1747, 1193},{810, 1719, 1133},{827, 1720, 1130},{844, 1727, 1116},{839, 1707, 1097},
{1062, 2011, 1106},{1081, 2032, 1138},{1108, 2058, 1168},{1108, 2084, 1226},{1123, 2083, 1210},{1112, 2078, 1209},{1103, 2037, 1132},{1103, 2043, 1154},
{770, 1227, 479},{775, 1226, 481},{772, 1229, 473},{813, 1220, 502},{752, 1043, 477},{556, 815, 426},{655, 934, 429},{492, 1028, 381}
};
int test_data_output[test_data_num][target_num] = {
{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},
{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},
{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},
{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},
{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},
{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},
{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1}
};
	

	/******************************************************************
 * End Network Configuration
 ******************************************************************/
int ReportEvery10;
int RandomizedIndex[train_data_num];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

float data_mean[3] ={0};
float data_std[3] ={0};

float Hidden[HiddenNodes];
float Output[target_num];
float HiddenWeights[input_length+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][target_num];
float HiddenDelta[HiddenNodes];
float OutputDelta[target_num];
float ChangeHiddenWeights[input_length+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][target_num];

int target_value;
int out_value;
int max;


/*---------------------------------------------------------------------------------------------------------*/
/* Define Function Prototypes                                                                              */
/*---------------------------------------------------------------------------------------------------------*/
void SYS_Init(void);
void UART0_Init(void);
void AdcSingleCycleScanModeTest(void);


void SYS_Init(void)
{
    /*---------------------------------------------------------------------------------------------------------*/
    /* Init System Clock                                                                                       */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Enable Internal RC 22.1184MHz clock */
    CLK_EnableXtalRC(CLK_PWRCON_OSC22M_EN_Msk);

    /* Waiting for Internal RC clock ready */
    CLK_WaitClockReady(CLK_CLKSTATUS_OSC22M_STB_Msk);

    /* Switch HCLK clock source to Internal RC and HCLK source divide 1 */
    CLK_SetHCLK(CLK_CLKSEL0_HCLK_S_HIRC, CLK_CLKDIV_HCLK(1));

    /* Enable external XTAL 12MHz clock */
    CLK_EnableXtalRC(CLK_PWRCON_XTL12M_EN_Msk);

    /* Waiting for external XTAL clock ready */
    CLK_WaitClockReady(CLK_CLKSTATUS_XTL12M_STB_Msk);

    /* Set core clock as PLL_CLOCK from PLL */
    CLK_SetCoreClock(PLL_CLOCK);

    /* Enable UART module clock */
    CLK_EnableModuleClock(UART0_MODULE);

    /* Enable ADC module clock */
    CLK_EnableModuleClock(ADC_MODULE);

    /* Select UART module clock source */
    CLK_SetModuleClock(UART0_MODULE, CLK_CLKSEL1_UART_S_PLL, CLK_CLKDIV_UART(1));

    /* ADC clock source is 22.1184MHz, set divider to 7, ADC clock is 22.1184/7 MHz */
    CLK_SetModuleClock(ADC_MODULE, CLK_CLKSEL1_ADC_S_HIRC, CLK_CLKDIV_ADC(7));

    /*---------------------------------------------------------------------------------------------------------*/
    /* Init I/O Multi-function                                                                                 */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Set GPB multi-function pins for UART0 RXD and TXD */
    SYS->GPB_MFP &= ~(SYS_GPB_MFP_PB0_Msk | SYS_GPB_MFP_PB1_Msk);
    SYS->GPB_MFP |= SYS_GPB_MFP_PB0_UART0_RXD | SYS_GPB_MFP_PB1_UART0_TXD;

    /* Disable the GPA0 - GPA3 digital input path to avoid the leakage current. */
    GPIO_DISABLE_DIGITAL_PATH(PA, 0xF);

    /* Configure the GPA0 - GPA3 ADC analog input pins */
    SYS->GPA_MFP &= ~(SYS_GPA_MFP_PA0_Msk | SYS_GPA_MFP_PA1_Msk | SYS_GPA_MFP_PA2_Msk | SYS_GPA_MFP_PA3_Msk) ;
    SYS->GPA_MFP |= SYS_GPA_MFP_PA0_ADC0 | SYS_GPA_MFP_PA1_ADC1 | SYS_GPA_MFP_PA2_ADC2 | SYS_GPA_MFP_PA3_ADC3 ;
    SYS->ALT_MFP1 = 0;

}

/*---------------------------------------------------------------------------------------------------------*/
/* Init UART                                                                                               */
/*---------------------------------------------------------------------------------------------------------*/
void UART0_Init()
{
    /* Reset IP */
    SYS_ResetModule(UART0_RST);

    /* Configure UART0 and set UART0 Baudrate */
    UART_Open(UART0, 115200);
}

void scale_data()
{
		float sum[3] = {0};
		int i, j;
		
		// Compute Data Mean
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length; j++){
				sum[j] += train_data_input[i][j];
			}
		}
		for(j = 0; j < input_length ; j++){
			data_mean[j] = sum[j] / train_data_num;
			printf("MEAN: %.2f\n", data_mean[j]);
			sum[j] = 0.0;
		}
		
		// Compute Data STD
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length ; j++){
				sum[j] += pow(train_data_input[i][j] - data_mean[j], 2);
			}
		}
		for(j = 0; j < input_length; j++){
			data_std[j] = sqrt(sum[j]/train_data_num);
			printf("STD: %.2f\n", data_std[j]);
			sum[j] = 0.0;
		}
}

void normalize(float *data)
{
		int i;
	
		for(i = 0; i < input_length; i++){
			data[i] = (data[i] - data_mean[i]) / data_std[i];
		}
}

int train_preprocess()
{
    int i;
    
    for(i = 0 ; i < train_data_num ; i++)
    {
        normalize(train_data_input[i]);
    }
		
    return 0;
}

int test_preprocess()
{
    int i;

    for(i = 0 ; i < test_data_num ; i++)
    {
        normalize(test_data_input[i]);
    }
		
    return 0;
}

int data_setup()
{
    int i;
		//int j;
		int p, ret;
		unsigned int seed = 1;
	
		/* Set the ADC operation mode as single-cycle, input mode as single-end and
                 enable the analog input channel 0, 1 and 2 */
    ADC_Open(ADC, ADC_ADCR_DIFFEN_SINGLE_END, ADC_ADCR_ADMD_SINGLE_CYCLE, 0x7);

    /* Power on ADC module */
    ADC_POWER_ON(ADC);

    /* Clear the A/D interrupt flag for safe */
    ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);

    /* Start A/D conversion */
    ADC_START_CONV(ADC);

    /* Wait conversion done */
    while(!ADC_GET_INT_FLAG(ADC, ADC_ADF_INT));
		
		for(i = 0; i < 3; i++)
    {
				seed *= ADC_GET_CONVERSION_DATA(ADC, i);
    }
		seed *= 1000;
		printf("\nRandom seed: %d\n", seed);
    srand(seed);

    ReportEvery10 = 1;
    for( p = 0 ; p < train_data_num ; p++ ) 
    {    
        RandomizedIndex[p] = p ;
    }
		
		scale_data();
    ret = train_preprocess();
    ret |= test_preprocess();
    if(ret) //Error Check
        return 1;

    return 0;
}

void run_train_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Train result:\n");
    for( p = 0 ; p < train_data_num ; p++ )
    { 
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / train_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);
}

void run_test_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Test result:\n");
    for( p = 0 ; p < test_data_num ; p++ )
    { 
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (test_data_output[p][i] > test_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += test_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / test_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);
}

float Get_Train_Accuracy()
{
    int i, j, p;
    int correct = 0;
		float accuracy = 0;
    for (p = 0; p < train_data_num; p++)
    {
/******************************************************************
* Compute hidden layer activations
******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

/******************************************************************
* Compute output layer activations
******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        //get target value
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        //get output value
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;
        //compare output and target
        if (out_value==target_value)
        {
            correct++;
        }
    }

    // Calculate accuracy
    accuracy = (float)correct / train_data_num;
    return accuracy;
}

void load_weight()
{
    int i,j;
    printf("\n=======Hidden Weight=======\n");
    printf("{");
    for(i = 0; i <= input_length ; i++)
    {
        printf("{");
        for (j = 0; j < HiddenNodes; j++)
        {
            if(j!=HiddenNodes-1){
                printf("%f,", HiddenWeights[i][j]);
            }else{
                printf("%f", HiddenWeights[i][j]);
            }
        }
        if(i!=input_length){
            printf("},\n");
        }else {
            printf("}");
        }
    }
    printf("}\n");

    printf("\n=======Output Weight=======\n");

    for(i = 0; i <= HiddenNodes ; i++)
    {
        printf("{");
        for (j = 0; j < target_num; j++)
        {
            if(j!=target_num-1){
                printf("%f,", OutputWeights[i][j]);
            }else{
                printf("%f", OutputWeights[i][j]);
            }
        }
        if(i!=HiddenNodes){
            printf("},\n");
        }else {
            printf("}");
        }
    }
    printf("}\n");
}

void AdcSingleCycleScanModeTest()
{
		int i, j;
    uint32_t u32ChannelCount;
    float single_data_input[3];
		char output_string[10] = {NULL};

    printf("\n");	
		printf("[Phase 3] Start Prediction ...\n\n");
		PB2=1;
    while(1)
    {
			
				/* Set the ADC operation mode as single-cycle, input mode as single-end and
                 enable the analog input channel 0, 1, 2 and 3 */
        ADC_Open(ADC, ADC_ADCR_DIFFEN_SINGLE_END, ADC_ADCR_ADMD_SINGLE_CYCLE, 0x7);

        /* Power on ADC module */
        ADC_POWER_ON(ADC);

        /* Clear the A/D interrupt flag for safe */
        ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);

        /* Start A/D conversion */
        ADC_START_CONV(ADC);

        /* Wait conversion done */
        while(!ADC_GET_INT_FLAG(ADC, ADC_ADF_INT));

        for(u32ChannelCount = 0; u32ChannelCount < 3; u32ChannelCount++)
        {
            single_data_input[u32ChannelCount] = ADC_GET_CONVERSION_DATA(ADC, u32ChannelCount);
        }
				normalize(single_data_input);
						

				// Compute hidden layer activations
				for( i = 0 ; i < HiddenNodes ; i++ ) {    
						Accum = HiddenWeights[input_length][i] ;
						for( j = 0 ; j < input_length ; j++ ) {
								Accum += single_data_input[j] * HiddenWeights[j][i] ;
						}
						Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
				}

				// Compute output layer activations
				for( i = 0 ; i < target_num ; i++ ) {    
						Accum = OutputWeights[HiddenNodes][i] ;
						for( j = 0 ; j < HiddenNodes ; j++ ) {
								Accum += Hidden[j] * OutputWeights[j][i] ;
						}
						Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
				}
						
				max = 0;
				for (i = 1; i < target_num; i++) 
				{
						if (Output[i] > Output[max]) {
								max = i;
						}
				}
				out_value = max;
				
				switch(out_value){
						case 0:
								strcpy(output_string, "Background");
								break;
						case 1:
								strcpy(output_string, "Blue");	
								break;
						case 2:
								strcpy(output_string, "Magenta");	
								break;
						case 3:
								strcpy(output_string, "Red");
								break;
						case 4:
								strcpy(output_string, "Orange");	
								break;
						case 5:
								strcpy(output_string, "Yellow");	
								break;
						case 6:
								strcpy(output_string, "Green");
								break;
				}
				
				printf("\rPrediction output: %-10s", output_string);
				CLK_SysTickDelay(500000);


    }
}

/*---------------------------------------------------------------------------------------------------------*/
/* MAIN function                                                                                           */
/*---------------------------------------------------------------------------------------------------------*/

int main(void)
{
		int i, j, p, q, r;
    float accuracy=0;

    /* Unlock protected registers */
    SYS_UnlockReg();

    /* Init System, IP clock and multi-function I/O */
    SYS_Init();

    /* Lock protected registers */
    SYS_LockReg();

    /* Init UART0 for printf */
    UART0_Init();
	
	  GPIO_SetMode(PB, BIT2, GPIO_PMD_OUTPUT);
	  PB2=0;
	
		printf("\n+-----------------------------------------------------------------------+\n");
    printf("|                        Machine Learning                        |\n");
    printf("+-----------------------------------------------------------------------+\n");
		printf("System clock rate: %d Hz\n", SystemCoreClock);

    printf("\n[Phase 1] Initialize DataSet ...");
	  /* Data Init (Input / Output Preprocess) */
		if(data_setup()){
        printf("[Error] Datasets Setup Error\n");
        return 0;
    }else
				printf("Done!\n\n");
		
		printf("[Phase 2] Start Model Training ...\n");
		// Initialize HiddenWeights and ChangeHiddenWeights 
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
        for( j = 0 ; j <= input_length ; j++ ) { 
            ChangeHiddenWeights[j][i] = 0.0 ;
            Rando = (float)((rand() % 100))/100;
            HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Initialize OutputWeights and ChangeOutputWeights
    for( i = 0 ; i < target_num ; i ++ ) {    
        for( j = 0 ; j <= HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = 0.0 ;  
            Rando = (float)((rand() % 100))/100;        
            OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Begin training 
    for(TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++)
    {
        Error = 0.0 ;

        // Randomize order of training patterns
        for( p = 0 ; p < train_data_num ; p++) {
            q = rand()%train_data_num;
            r = RandomizedIndex[p] ; 
            RandomizedIndex[p] = RandomizedIndex[q] ; 
            RandomizedIndex[q] = r ;
        }

        // Cycle through each training pattern in the randomized order
        for( q = 0 ; q < train_data_num ; q++ ) 
        {    
            p = RandomizedIndex[q];

            // Compute hidden layer activations
            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = HiddenWeights[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) {
                    Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
                }
                Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
            }

            // Compute output layer activations and calculate errors
            for( i = 0 ; i < target_num ; i++ ) {    
                Accum = OutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    Accum += Hidden[j] * OutputWeights[j][i] ;
                }
                Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
                OutputDelta[i] = (train_data_output[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
                Error += 0.5 * (train_data_output[p][i] - Output[i]) * (train_data_output[p][i] - Output[i]) ;
            }

            // Backpropagate errors to hidden layer
            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = 0.0 ;
                for( j = 0 ; j < target_num ; j++ ) {
                    Accum += OutputWeights[i][j] * OutputDelta[j] ;
                }
                HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
            }

            // Update Input-->Hidden Weights
            for( i = 0 ; i < HiddenNodes ; i++ ) {     
                ChangeHiddenWeights[input_length][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[input_length][i] ;
                HiddenWeights[input_length][i] += ChangeHiddenWeights[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) { 
                    ChangeHiddenWeights[j][i] = LearningRate * train_data_input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
                    HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
                }
            }

            // Update Hidden-->Output Weights
            for( i = 0 ; i < target_num ; i ++ ) {    
                ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
                OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
                    OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
                }
            }
        }
        accuracy = Get_Train_Accuracy();

        // Every 10 cycles send data to terminal for display
        ReportEvery10 = ReportEvery10 - 1;
        if (ReportEvery10 == 0)
        {
            
            printf ("\nTrainingCycle: %ld\n",TrainingCycle);
            printf ("Error = %.5f\n",Error);
            printf ("Accuracy = %.2f /100 \n",accuracy*100);
            //run_train_data();

            if (TrainingCycle==1)
            {
                ReportEvery10 = 9;
            }
            else
            {
                ReportEvery10 = 10;
            }
        }

        // If error rate is less than pre-determined threshold then end
        if( accuracy >= goal_acc ) break ;
    }

    printf ("\nTrainingCycle: %ld\n",TrainingCycle);
    printf ("Error = %.5f\n",Error);
    run_train_data();
    printf ("Training Set Solved!\n");
    printf ("--------\n"); 
    printf ("Testing Start!\n ");
    run_test_data();
    printf ("--------\n"); 
    ReportEvery10 = 1;
    load_weight();
		
		printf("\nModel Training Phase has ended.\n");

    /* Start prediction */
    AdcSingleCycleScanModeTest();

    while(1);
}