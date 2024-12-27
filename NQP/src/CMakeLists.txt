target_sources(qnnls PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/log.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NNLSQPSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/decorators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linSolvers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scaler.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/log.h
    ${CMAKE_CURRENT_SOURCE_DIR}/NNLSQPSolver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/timers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/decorators.h
    ${CMAKE_CURRENT_SOURCE_DIR}/linSolvers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scaler.h

)
