class Model {
public:
    Model(String boltModelFilePath, String modelHeaderFilePath)
    {
        parseBoltModel(boltModelFilePath);
    }

    ~Model()
    {
        // destroy model;
        // TODO
    }

    // load an bolt model file into memory
    void parseBoltModel(String modelPath)
    {
        // use model_tools deserialize function
        // TODO
    }

    // write bolt model weight into model.h/model.c in the array format
    // model.h
    // char *weight0 = [0, 1, 2, 3];
    // output = {
    //   op_name: weight0
    // }
    pair<OperatorParameter *, std::map<String, String>> generate()
    {}

private:
    ModelSpec modelSpec;
};
