#pragma once

#include <string>
#include <External/AnyOption/anyoption.h>
#include <Utils/Chrono.cuh>
#include <Utils/TypeAlias.h>

class Options
{

    // Members
    public:
    bool statistics;
    u32 queueSize;
    u32 timeout;
    u32 widthCpu;
    u32 mddsCpu;
    u32 widthGpu;
    u32 mddsGpu;
    float probEq;
    float probNeq;
    u32 randomSeed;
    char const * inputFilename;
    private:
    AnyOption* const anyOption;

    // Functions
    public:
    Options();
    ~Options();
    bool parseOptions(int argc, char* argv[]);
    void printUsage() const;
    void printOptions();
};

Options::Options() :
    statistics(false),
    inputFilename(nullptr),
    queueSize(50000),
    timeout(0),
    widthCpu(0),
    mddsCpu(0),
    widthGpu(0),
    mddsGpu(0),
    probEq(0),
    probNeq(0),
    randomSeed(static_cast<u32>(Chrono::now() % 1000)),
    anyOption(new AnyOption())
{
    // Help
    anyOption->addUsage("Usage: ");
    anyOption->addUsage("");
    anyOption->addUsage(" -h --help             Print this help");
    anyOption->addUsage(" -s                    Print search statistics");
    //anyOption->addUsage(" -q <integer>          Size of the queue for the initial search");
    anyOption->addUsage(" -t <integer>          Seconds of timeout");
    anyOption->addUsage(" --wc <integer>        Width of MDDs explored on CPU");
    anyOption->addUsage(" --mc <integer>        Number of MDDs explored on CPU");
    anyOption->addUsage(" --wg <integer>        Width of MDDs explored on GPU");
    anyOption->addUsage(" --mg <integer>        Number of MDDs explored on GPU");
    anyOption->addUsage(" --eq <float>          Probability to use a value in a neighborhood during LNS");
    anyOption->addUsage(" --neq <float>         Probability to discard a value in a neighborhood during LNS");
    anyOption->addUsage(" --rs <integer>        Random seed");
    anyOption->addUsage("");

    anyOption->setFlag("help",'h');
    anyOption->setFlag('s');
    //anyOption->setOption('q');
    anyOption->setOption('t');
    anyOption->setOption("wc");
    anyOption->setOption("mc");
    anyOption->setOption("wg");
    anyOption->setOption("mg");
    anyOption->setOption("eq");
    anyOption->setOption("neq");
    anyOption->setOption("rs");

}

Options::~Options()
{
     free(anyOption);
}


bool Options::parseOptions(int argc, char* argv[])
{
    anyOption->processCommandArgs(argc, argv);

    if (not anyOption->hasOptions())
    {
        printUsage();
        return false;
    }

    if (anyOption->getFlag('h') or anyOption->getFlag("help"))
    {
        anyOption->printUsage();
        return false;
    }

    if (anyOption->getFlag('s'))
        statistics = true;

    /*
    if (anyOption->getValue('q') != nullptr)
        queueSize = static_cast<u32>(std::stoi(anyOption->getValue('q')));
    else
        return false;
    */

    if (anyOption->getValue('t') != nullptr)
        timeout = static_cast<u32>(std::stoi(anyOption->getValue('t')));
    else
        return false;

    if (anyOption->getValue("wc") != nullptr)
        widthCpu = static_cast<u32>(std::stoi(anyOption->getValue("wc")));
    else
        return false;

    if (anyOption->getValue("mc") != nullptr)
        mddsCpu = static_cast<u32>(std::stoi(anyOption->getValue("mc")));
    else
        return false;

    if (anyOption->getValue("wg") != nullptr)
        widthGpu = static_cast<u32>(std::stoi(anyOption->getValue("wg")));
    else
        return false;


    if (anyOption->getValue("mg") != nullptr)
        mddsGpu = static_cast<u32>(std::stoi(anyOption->getValue("mg")));
    else
        return false;

    if (anyOption->getValue("eq") != nullptr)
        probEq = static_cast<float>(std::stof(anyOption->getValue("eq")));
    else
        return false;

    if (anyOption->getValue("neq") != nullptr)
        probNeq = static_cast<float>(std::stof(anyOption->getValue("neq")));
    else
        return false;

    if (anyOption->getValue("rs") != nullptr)
        randomSeed = static_cast<u32>(std::stoi(anyOption->getValue("rs")));
    else
        return false;

    if (anyOption->getArgv(0) != nullptr)
        inputFilename = anyOption->getArgv(0);
    else
        return false;

    return true;
}
void Options::printOptions()
{
    printf("[INFO] Input file: %s\n", inputFilename);
    //printf("[INFO] Queue size: %u\n", queueSize);
    printf("[INFO] Timeout: %u\n", timeout);
    printf("[INFO] CPU: Width %u | MDDs %u\n", widthCpu, mddsCpu);
    printf("[INFO] GPU: Width %u | MDDs %u\n", widthGpu, mddsGpu);
    printf("[INFO] LNS: = %.3f | ≠ %.3f | Random seed %u\n", probEq, probNeq, randomSeed);
}

void Options::printUsage() const
{
    anyOption->printUsage();
}
