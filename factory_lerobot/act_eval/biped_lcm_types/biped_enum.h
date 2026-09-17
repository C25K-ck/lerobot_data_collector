#ifndef BIPED_ENUM_H
#define BIPED_ENUM_H

const std::string LCM_RESPONSE_NAME = "lcm_response"; // lcm port 7670
const std::string LCM_COMMAND_NAME = "lcm_request"; // lcm port 7671


typedef enum FSM_StateEnum {
    K_OFF             = -1,
    K_Passive         = 0,
    K_PureDamper      = 10,
    K_RecoveryStand   = 1,
    K_BalanceStand    = 2,
    K_Walk            = 3,
    K_Swing            = 4,//hang for debug
    K_OneLegStance    = 5,
    K_TurnWaist    = 6,
    K_WaistTurn       = 7,


    K_ZMP = 20,
    K_OnlineZMP = 21,

    K_WalkCMP = 30,
    K_BalanceStandW = 31,
    K_WalkCMP2 = 32,
    K_WalkCMP3 = 33,

    K_Test = 40,

    K_Manipulation = 70,
    K_OfflineWaving = 71,
    K_OfflineCoffee = 72,

    K_PDIK  = 99,

} FSM_StateEnum;


#endif //BIPED_ENUM_H
