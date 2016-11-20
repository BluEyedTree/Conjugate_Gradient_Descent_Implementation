
import numpy
    
def logistic(x, theta):
    """
    This is the logistic function, which is how we are modeling our probability
    that a student passes a test given the number of hours studied,
    
    Arguments:
    ----------
    x -- array of lists that are [1, number of hours studied for student i] for all students
    theta -- array of parameters
    """
    return 1 / (1 + numpy.exp(-x.dot(theta)))

def loglikeFunc(theta):
    """
    Our objective function:
    The negative of the log likelihood function.
    The log likelihood function is the product of the log of the logistic function 
    for all that passed plus the product of the log of (1 - the logistic function)
    for all that failed, evaluated with a given theta (the set of parameters of the 
    logistic function)
    
    Arguments:
    ----------
    theta -- array of parameters
    """
    global count_like_calls
    count_like_calls += 1
    t = numpy.array([[1,0.50], [1,0.75], [1,1.00], [1,1.25],
                        [1,1.50], [1,1.75], [1,1.75], [1,2.00],
                        [1,2.25], [1,2.50], [1,2.75], [1,3.00],
                        [1,3.25], [1,3.50], [1,4.00], [1,4.25],
                        [1,4.50], [1,4.75], [1,5.00], [1,5.50]])                        
    p = numpy.array([0, 0, 0, 0,
                        0, 0, 1, 0,
                        1, 0, 1, 0,
                        1, 0, 1, 1,
                        1, 1, 1, 1])
    ppass = sum([numpy.log(logistic(t[i],theta)) if p[i] == 1 else 0 for i in range(len(t))])
    pfail = sum([numpy.log(1 - logistic(t[i],theta)) if p[i] == 0 else 0 for i in range(len(t))])
    return - ppass - pfail


def gradFunc(theta):
    """
    Computes the gradient of the above function. Returns an array of
    values, with the first value being the derivative in the theta_0 direction, and
    the second being the derivative in the theta_1 direction

    Arguments
    ---------
    theta -- array of floats; a point at which the gradient is computed
    """
    global count_grad_calls
    count_grad_calls += 1
    t = numpy.array([[1,0.50], [1,0.75], [1,1.00], [1,1.25],
                        [1,1.50], [1,1.75], [1,1.75], [1,2.00],
                        [1,2.25], [1,2.50], [1,2.75], [1,3.00], 
                        [1,3.25], [1,3.50], [1,4.00], [1,4.25],
                        [1,4.50], [1,4.75], [1,5.00], [1,5.50]])
                            
    p = numpy.array([0, 0, 0, 0,
                        0, 0, 1, 0,
                        1, 0, 1, 0,
                        1, 0, 1, 1,
                        1, 1, 1, 1])
    
    return numpy.dot(t.T,logistic(t,theta) - p)


def phi(alpha, theta):
    """
    1D slice of the 2D surface of loglikeFunc. Used in the line search method to
    select the correct step size
    
    Arguments
    ---------
    alpha -- step size; the parameter to be optimized later
    theta -- point in 2D theta space; array of 2 floats
    """
    global count_phi_calls
    global beta_attempts
    count_phi_calls += 1
    rho = -gradFunc(theta)
    beta_attempts.append(theta + alpha*rho)
    return loglikeFunc(theta + alpha*rho)


def dphi(alpha, theta):
    """
    Derivative of 1D slice function phi. Used in zoom function as well as line
    search
    
    Arguments
    ---------
    alpha -- step size; the parameter to be optimized later
    theta -- point in 2D theta space; array of 2 floats
    """
    global count_dphi_calls
    count_dphi_calls += 1
    rho = -gradFunc(theta)
    return numpy.dot(gradFunc(theta + alpha*rho), rho)


def golden(a, b, theta, tol=1e-4):
    """
    1D optimizer used to find the minimum of the phi function. Used in the line
    search function to help find correct step sizes.

    Arguments
    ---------
    a, b -- bounding values of alpha
    theta -- point in 2D theta space; array of 2 floats
    tol -- tolerance in error; returns a value when this tolerance is met
    """
    gr = (numpy.sqrt(5) + 1)/2
    c = b - ( b - a ) / gr
    d = a + (b-a) /gr
    while abs(c - d) > tol:
        if phi(c, theta) < phi(d, theta):
            b = d
        else:
            a = c
        c = b - (b-a) / gr
        d = a + (b-a) / gr
    return (b + a)/2


def zoom(a_lo, a_hi, theta, c1=1e-4, c2=0.1):
    """
   
    
    Arguments
    ---------
    a_lo, a_hi -- bounding values of alpha
    theta -- point in 2D theta space; array of 2 floats
    c1, c2 -- coefficient from 1st & 2nd Wolfe condition
    """
    counter = 0
    phi0 = phi(0, theta)
    dphi0 = dphi(0, theta)
    while counter < 1000:
        counter += 1
        a_j = golden(a_lo, a_hi, theta)
        phi_j = phi(a_j, theta)
        con1 = (phi_j > phi0 +c1*a_j*dphi0)
        con2 = phi(a_j, theta) >= phi(a_lo, theta)
        if con1 or con2:
            a_hi = a_j
        else:
            dphi_j = dphi(a_j, theta)
            
            if abs(dphi_j) <= -c2*dphi0:
                return a_j
            if dphi_j*(a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_j


def line_search(theta, c1=1e-4, c2=0.1, a0 = 0, amax = 50):
    """
    The line_search method for finding the correct step size in our desired
    step direction.

    Arguments
    ---------
    theta -- point in 2D theta space; array of 2 floats
    c1, c2 -- coefficient from 1st & 2nd Wolfe condition
    a0, amax -- initial bounding alpha values
    """
    i = 1
    ai = 1
    a_old = a0
    phi0 = phi(0, theta)
    dphi0 = dphi(0, theta)
    while i < 1000:
        i += 1
        phi_i = phi(ai, theta)
        phi_old = phi(a_old, theta)
        con1 = phi_i > phi0 + c1*ai*dphi0
        con2 = phi_i >= phi_old
        if con1 or (con2 and i > 1):
            return zoom(a_old, ai, theta)
        dphi_i = dphi(ai, theta)
        if abs(dphi_i) <= -c2*dphi0:
            return ai
        if dphi_i >= 0:
            return zoom(ai, a_old, theta)
        ai, a_old = ai*2, ai


def gradient_descent(f, gradf, theta, tol = 1e-8):
    """
    Gradient Descent method. Selects a direction using the gradFunc, and then
    takes a step size alpha (determined by line search) in that direction.

    Arguments
    ---------
    f -- function to optimize
    gradf -- gradient of the function
    theta -- point in 2D theta space; array of 2 floats
    """
    global count_phi_calls
    global count_dphi_calls
    global count_like_calls
    global count_grad_calls
    global beta_attempts
    
    beta_attempts = []
    count_phi_calls = 0
    count_dphi_calls = 0
    count_like_calls = 0
    count_grad_calls = 0

    error = f(theta)
    counter = 0
    theta_list = [theta]
    rho = -gradf(theta)
    while error > tol and counter < 1000:
        a_i = line_search(theta)
        theta_new = theta + a_i * rho
        gradf_old, gradf_new = gradf(theta), gradf(theta_new)
        #error = abs(f(theta_new) - f(theta))
        error = abs(numpy.dot(gradf_new,gradf_new))
        if counter%2 == 0:
            beta = 0
        else:
            beta = numpy.dot(gradf_new.T,gradf_new)/numpy.dot(gradf_old.T,gradf_old)
        rho = -gradf_new + beta * rho
        theta_list.append(theta_new)
        theta = theta_new
        counter += 1
    print("Number of line search function calls: ", counter - 1)
    print("Number of phi function calls: ", count_phi_calls)
    print("Number of dphi function calls: ", count_dphi_calls)
    print("Number of objective function calls: ", count_like_calls)
    print("Number of gradient function calls: ", count_grad_calls)     
    print("Final Beta Values: ", theta_new)
    

    
    return theta_new, theta_list


if __name__ == '__main__':
    """
    quick clarification:
    The betas in this section were changed to thetas in the rest of the 
    functions to distinguish them from the beta value used in the conjugate gradient 
    descent algorithm.  I have them as betas here because this is what we have been
    calling the parameters in all of the previous algorithms for this model.
    """
    global beta_attempts
    
    from matplotlib import pyplot
    
    beta = numpy.array([0, 0])
    b, blist = gradient_descent(loglikeFunc, gradFunc, beta)
    
    
    # plot all the betas
    size = 10
    pyplot.figure(figsize=(size, size))
    b0 = [ i for i,j in blist ]
    b1 = [ j for i,j in blist ]
    garbageb0 = [ i for i,j in beta_attempts ]
    garbageb1 = [ j for i,j in beta_attempts ]
    pyplot.plot(garbageb0,garbageb1, 'go', label="$\\beta$ Pair Attempts")
    pyplot.plot(b0, b1, 'ro', label="$\\beta$ Pairs")
    pyplot.plot(b0, b1, label="Line Showing Step Sizes")
    pyplot.legend(loc=0)
    pyplot.xlabel('$\\beta_0$', fontsize = 20)
    pyplot.ylabel('$\\beta_1$', fontsize = 20)
    pyplot.savefig("beta_plot_conj_ML.png")
    pyplot.show()