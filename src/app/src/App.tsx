import { BrowserRouter, Route, Routes } from "react-router-dom"
import HomePage from "./pages/HomePage"
import AddTaskPage from "./pages/AddTaskPage"
import PendingTasksPage from "./pages/PendingTasksPage"

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/task" element={<AddTaskPage />} />
        <Route path="/pending" element={<PendingTasksPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
