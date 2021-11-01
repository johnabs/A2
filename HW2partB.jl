# To run this, you need to install Julia 1.6, and install the packages listed on the next line.
# This can be done by pressing "]" and typing add *package name*, then backspace to return to
# the julia REPL.
using Evolutionary, CSV, DataFrames, StatsPlots

function getInstance(file)
    data=CSV.read(file,DataFrame,skipto=6,header=false) #imports data from CSV file into dataframe

    #Gets constraint value
    con=open(file) do file
        fileText= readlines(file)
        con=fileText[3][3:end]
        return parse(Int,con)
    end

    #Create variables, knapsack options (n), weights (w), and values (v)
    n=size(data)[1]
    w=data[:,3]
    v=data[:,2]
    return con,n,w,v
end

function bestPartA(con,n,w,v,popSize)
    
    function x0(n)
        return flip(BitVector(zeros(n)))
    end
    
    mutRate = (8/9)*((n-1)/(popSize*n)) #From part a
    crossRate = 0.5 #From part a
    
    #Create objective function, which defaults to a dynamic penalty if it violates the constraint,
    #and the sum is set to negative as this EA uses minimization
    f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : (sum(w.*x)-con))
    
    ga = GA(populationSize=popSize,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=mutRate,crossoverRate=crossRate,ɛ = 0.1);
    gaResults = Evolutionary.optimize(f, x0(n), ga, Evolutionary.Options(iterations=10000))
    
    return gaResults #technically unecessary, but helps me think
end

function partb(file)
    #This function accepts a filename and returns a dataframe of optimization results
    #The returned dataframe has columns for algorithm/approach type, population size, and solution fitness value
    
    # Initialization
    con,n,w,v = getInstance(file)
    popSize = [n,3*n,5*n,7*n,9*n,11*n,13*n]
    f(x) = -sum(v.*x) #objective function, made negative to convert to minimization
    runs = 30 #number of runs per population size

    function feasibleRand(currentGA)
        #This function takes the result of a GA optimzation and randomly generates
        # as many feasible solutions as function calls reported by the GA
        function oneSol(x)
            #This function creates a single feasible solution from an initially
            # empty knapsack
            weight = sum(w.*x) #initialize constraint value
            while weight < con
                item = rand(findall(x->x==0,x)) #randomly selects an item not currently in knapsack
                x[item] = true #adds item to knapsack
                if sum(w.*x) > con #removes item if it would violate the constraint
                    x[item] = false
                    weight = con + 1 #sets weight to exit loop
                else
                    weight = sum(w.*x) #updates weight normally
                end
            end
            return x
        end

        #Create feasible solutions
        numSolns = Evolutionary.f_calls(currentGA)
        solutions = oneSol.([BitVector(zeros(n)) for _ in 1:numSolns])
        
        #Evaluate solutions and finds best result
        bestFitness = minimum(f.(solutions))

        return bestFitness
    end
    
    function truRand(currentGA)
        #Randomly generates solutions
        numSolns = Evolutionary.f_calls(currentGA)
        solutions = [BitVector(rand([0 1],n)) for _ in 1:numSolns]
        
        weights = [sum(w.*x) for x in solutions] #calculates constraint values for each solution
        infeasible = findall(weights -> weights > con, weights) #finds position of infeasible solutions
        fitness = f.(solutions) #evaluates solutions
        fitness[infeasible] .= 0 #sets infeasible solutions to zero

        bestFitness = minimum(fitness)
        return bestFitness
    end

    #pre-allocating space for storing algorithm results
    gaBests = zeros(runs, length(popSize))
    frandBests = zeros(runs, length(popSize))
    trandBests = zeros(runs,length(popSize))
    z = 1
    
    #executes algorithms for each population size, 30 runs per size 
    for i in popSize  
        for j in 1:runs
            currentGA = bestPartA(con,n,w,v,i)
            gaBests[j,z] = -Evolutionary.minimum(currentGA)
            frandBests[j,z] = -feasibleRand(currentGA)
            trandBests[j,z] = -truRand(currentGA)
        end
        z +=1
    end
    
    #builds dataframe of all results
    ga_df = DataFrame(alg = fill("GA (Best Part A)",runs*length(popSize)), Npop = repeat(popSize,inner=runs), fitness = gaBests[:])
    frand_df = DataFrame(alg = fill("RS (Feasible-only)",runs*length(popSize)), Npop = repeat(popSize,inner=runs), fitness = frandBests[:])
    trand_df = DataFrame(alg = fill("RS (All)",runs*length(popSize)), Npop = repeat(popSize,inner=runs), fitness = trandBests[:])
    df = vcat(ga_df,frand_df,trand_df)

    return df
end

# all problem instances to be considered
filename = ["knapPI_11_100_1000.csv", "knapPI_13_50_1000.csv",
"knapPI_13_200_1000.csv","knapPI_14_50_1000.csv",
"knapPI_15_50_1000.csv","knapPI_16_50_1000.csv"]

#Executes partb over each problem instance, storing the resulting dataframes separately
dfOne,dfTwo,dfThree,dfFour,dfFive,dfSix = partb.(filename)

#Box plots of results, grouped by algorithm type, all but plotA to be used as subplots
plotA = @df dfOne groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Box Plot of Fitness vs Population Size for Instance 11_100_1000",legend=:outerleft, titlefont=(10))
plotB = @df dfTwo groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Instance 13_50_1000",legend=false, titlefont=(10))
plotC = @df dfThree groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Instance 13_200_1000",legend=false, titlefont=(10))
plotD = @df dfFour groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Instance 14_50_1000",legend=false, titlefont=(10))
plotE = @df dfFive groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Instance 15_50_1000",legend=false, titlefont=(10))
plotF = @df dfSix groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Instance 16_50_1000",legend=false, titlefont=(10))

#a pseudo-legend plot to act as a master legend for a plot of subplots
plotFiller = plot((1:3)',legend=true, framestyle=:none,label = ["GA (Best Part A)" "RS (Feasible-only)" "RS (All)"],title="Fitness vs Population Size")

#Plots with 2 or more subplots
plotC_E = (plotC,plotE,layout=(1,2), legend:outerbottom)
plotB_D_F = plot(plotB,plotD,plotF,layout = (2,2))

#Saves resulting images as a png for later reference
png(plotA,"GAvsRand_11_100")
png(plotC_E,"GAvsRand_13_15")
png(plotB_D_F,"GAvsRand_13_14_16")
