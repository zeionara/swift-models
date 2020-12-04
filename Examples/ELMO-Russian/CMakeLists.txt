add_executable(ELMO-Russian
        main.swift)
target_link_libraries(ELMO-Russian PRIVATE
        TextModels
        Datasets
        TrainingLoop)


install(TARGETS ELMO-Russian
        DESTINATION bin)
