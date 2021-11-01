using Evolutionary, CSV, DataFrames, StatsPlots

function getInstance(file)
    data=CSV.read(file,DataFrame,skipto=6,header=false)

    #Also read in constraint value
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
    con,n,w,v = getInstance(file)
    popSize = [n,3*n,5*n,7*n,9*n,11*n,13*n]
    f(x) = -sum(v.*x)
    runs = 30

    function feasibleRand(currentGA)
        function oneSol(x)
            weight = sum(w.*x)
            while weight < con
                item = rand(findall(x->x==0,x))
                x[item] = true
                if sum(w.*x) > con
                    x[item] = false
                    weight = con + 1
                else
                    weight = sum(w.*x)
                end
            end
            return x
        end
        numSolns = Evolutionary.f_calls(currentGA)
        solutions = oneSol.([BitVector(zeros(n)) for _ in 1:numSolns])
        bestFitness = minimum(f.(solutions))
        return bestFitness
    end
    
    function truRand(currentGA)
        numSolns = Evolutionary.f_calls(currentGA)
        solutions = [BitVector(rand([0 1],n)) for _ in 1:numSolns]
        weights = [sum(w.*x) for x in solutions]
        infeasible = findall(weights -> weights > con, weights)
        fitness = f.(solutions)
        fitness[infeasible] .= 0
        bestFitness = minimum(fitness)
        return bestFitness
    end
    
    gaBests = zeros(runs, length(popSize))
    frandBests = zeros(runs, length(popSize))
    trandBests = zeros(runs,length(popSize))
    z = 1
    for i in popSize  
        for j in 1:runs
            currentGA = bestPartA(con,n,w,v,i)
            gaBests[j,z] = -Evolutionary.minimum(currentGA)
            frandBests[j,z] = -feasibleRand(currentGA)
            trandBests[j,z] = -truRand(currentGA)
        end
        z +=1
    end
    
    ga_df = DataFrame(alg = fill("GA (Best Part A)",runs*length(popSize)), Npop = repeat(popSize,inner=runs), fitness = gaBests[:])
    frand_df = DataFrame(alg = fill("RS (Feasible-only)",runs*length(popSize)), Npop = repeat(popSize,inner=runs), fitness = frandBests[:])
    trand_df = DataFrame(alg = fill("RS (All)",runs*length(popSize)), Npop = repeat(popSize,inner=runs), fitness = trandBests[:])
    df = vcat(ga_df,frand_df,trand_df)

    return df
end

filename = ["knapPI_11_100_1000.csv", "knapPI_13_50_1000.csv",
"knapPI_13_200_1000.csv","knapPI_14_50_1000.csv",
"knapPI_15_50_1000.csv","knapPI_16_50_1000.csv"]

dfOne,dfTwo,dfThree,dfFour,dfFive,dfSix = partb.(filename)

plotA = @df dfOne groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Box Plot of Fitness vs Population Size",legend=:outerright)
plotB = @df dfTwo groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Box Plot of Fitness vs Population Size",legend=:outerright)
plotC = @df dfThree groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Box Plot of Fitness vs Population Size",legend=:outerright)
plotD = @df dfFour groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Box Plot of Fitness vs Population Size",legend=:outerright)
plotE = @df dfFive groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Box Plot of Fitness vs Population Size",legend=:outerright)
plotF = @df dfSix groupedboxplot(string.(:Npop), :fitness, group = :alg, title="Box Plot of Fitness vs Population Size",legend=:outerright)

png(plotA,"GAvsRand_11_100")
png(plotB,"GAvsRand_13_50")
png(plotC,"GAvsRand_13_200")
png(plotD,"GAvsRand_14_50")
png(plotE,"GAvsRand_15_50")
png(plotF,"GAvsRand_16_50")
