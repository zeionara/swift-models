add_executable(BERT-Russian
        main.swift)
target_link_libraries(BERT-Russian PRIVATE
        TextModels
        Datasets
        TrainingLoop)


install(TARGETS BERT-Russian
        DESTINATION bin)
